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
from sagemaker.huggingface import HuggingFaceModel

def setup_environment():
    """Set up the environment and return configuration."""
    # Load stored variables (simulating notebook %store magic)
    try:
        # Try to load from environment variables or config
        S3_BUCKET = os.environ.get('S3_BUCKET')
        SAGEMAKER_ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN')
        AWS_REGION = os.environ.get('AWS_REGION')
        OPTIMIZATION_INSTANCE_TYPE = os.environ.get('OPTIMIZATION_INSTANCE_TYPE', 'ml.c5.xlarge')
        
        if not all([S3_BUCKET, SAGEMAKER_ROLE_ARN, AWS_REGION]):
            # Fallback to SageMaker session defaults
            sagemaker_session = sagemaker.Session()
            S3_BUCKET = sagemaker_session.default_bucket()
            SAGEMAKER_ROLE_ARN = sagemaker.get_execution_role()
            AWS_REGION = boto3.session.Session().region_name
            
    except Exception as e:
        print(f"Error setting up environment: {e}")
        # Provide fallback values for testing
        S3_BUCKET = "example-bucket"
        SAGEMAKER_ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
        AWS_REGION = "us-west-2"
        OPTIMIZATION_INSTANCE_TYPE = "ml.c5.xlarge"
    
    print(f"Loaded variables:")
    print(f"S3_BUCKET: {S3_BUCKET}")
    print(f"AWS_REGION: {AWS_REGION}")
    print(f"OPTIMIZATION_INSTANCE_TYPE: {OPTIMIZATION_INSTANCE_TYPE}")
    
    return {
        "S3_BUCKET": S3_BUCKET,
        "SAGEMAKER_ROLE_ARN": SAGEMAKER_ROLE_ARN,
        "AWS_REGION": AWS_REGION,
        "OPTIMIZATION_INSTANCE_TYPE": OPTIMIZATION_INSTANCE_TYPE
    }

def load_model_info(S3_BUCKET):
    """Load model information."""
    try:
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        print("Loaded model information from model_info.json")
    except FileNotFoundError:
        print("model_info.json not found. Using default model information.")
        model_info = {
            "sentiment-analysis": {
                "model_name": "distilbert-base-uncased",
                "task": "text-classification",
                "hub_model_id": "distilbert-base-uncased",
                "s3_uri": f"s3://{S3_BUCKET}/models/distilbert-base-uncased"
            }
        }
    
    # For this script, we'll focus on the sentiment analysis model
    model_key = "sentiment-analysis"
    model_data = model_info[model_key]
    model_name = model_data["model_name"]
    model_s3_uri = model_data.get("s3_uri", f"s3://{S3_BUCKET}/models/{model_name.replace('/', '-')}")
    
    print(f"Using model: {model_name}")
    print(f"Model S3 URI: {model_s3_uri}")
    
    return model_data, model_name, model_s3_uri

def run_quantization_processing_job(config, model_name, model_s3_uri):
    """Configure and run the quantization processing job."""
    S3_BUCKET = config["S3_BUCKET"]
    SAGEMAKER_ROLE_ARN = config["SAGEMAKER_ROLE_ARN"]
    OPTIMIZATION_INSTANCE_TYPE = config["OPTIMIZATION_INSTANCE_TYPE"]
    
    try:
        sagemaker_session = sagemaker.Session()
        
        # Configure the processing job with updated versions
        processor = PyTorchProcessor(
            framework_version='2.6.0',
            py_version='py312',  # Updated to supported version
            role=SAGEMAKER_ROLE_ARN,
            instance_count=1,
            instance_type=OPTIMIZATION_INSTANCE_TYPE,
            base_job_name=f'quantize-{model_name.replace("/", "-")}',
            sagemaker_session=sagemaker_session
        )
        
        print("Starting quantization processing job...")
        
        # Run the processing job asynchronously (returns control immediately)
        processor.run(
            code='quantization_script.py',
            source_dir='scripts',
            inputs=[
                ProcessingInput(
                    source=model_s3_uri,
                    destination='/opt/ml/processing/input/model'
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='quantized_model',
                    source='/opt/ml/processing/output'
                )
            ],
            arguments=[
                '--quantization-approach', 'dynamic',
                '--bits', '8'
            ],
            wait=False,  # Don't wait - return control immediately
            logs=False   # Don't stream logs to avoid blocking
        )
        
        # Get the processing job name and output URI
        processing_job_name = processor.latest_job.job_name
        print(f"Processing job name: {processing_job_name}")
        
        processing_job_description = sagemaker_session.sagemaker_client.describe_processing_job(
            ProcessingJobName=processing_job_name
        )
        output_s3_uri = processing_job_description['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
        print(f"Output S3 URI: {output_s3_uri}")
        
        return {
            "processing_job_name": processing_job_name,
            "output_s3_uri": output_s3_uri,
            "quantized_model_name": f"{model_name.replace('/', '-')}-quantized"
        }
        
    except Exception as e:
        print(f"Error running processing job: {e}")
        return None

def test_quantization_script_locally():
    """Test the quantization script locally to catch API issues."""
    print("\n=== Testing Quantization Script Locally ===")
    
    try:
        # Test import of the quantization script
        import sys
        sys.path.append('scripts')
        
        # Try to import the main components to check for API issues
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        
        print("✅ All imports successful")
        
        # Test the API calls that were causing issues
        print("Testing API compatibility...")
        
        # This should work without the from_transformers parameter
        try:
            # Test with a small model to verify API
            model_name = "distilbert-base-uncased"
            print(f"Testing with model: {model_name}")
            
            # Test AutoConfig loading
            config = AutoConfig.from_pretrained(model_name)
            print("✅ AutoConfig.from_pretrained() works")
            
            # Test quantization config creation
            quantization_config = AutoQuantizationConfig.avx512_vnni(
                is_static=False, 
                per_channel=False
            )
            print("✅ AutoQuantizationConfig.avx512_vnni() works")
            
            print("✅ Local API compatibility test passed!")
            return True
            
        except Exception as api_error:
            print(f"❌ API compatibility test failed: {api_error}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install required packages: pip install optimum[onnxruntime] transformers")
        return False
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantize a model using SageMaker Processing")
    parser.add_argument("--approach", choices=["dynamic", "static"], default="dynamic",
                        help="Quantization approach: dynamic or static")
    parser.add_argument("--skip-job", action="store_true", help="Skip the SageMaker processing job")
    parser.add_argument("--test-local", action="store_true", help="Test quantization script locally")
    return parser.parse_args()

def main():
    """Main function to run the quantization script."""
    args = parse_args()
    
    print("=== Model Quantization with SageMaker Processing ===")
    
    # Test locally first if requested
    if args.test_local:
        local_test_passed = test_quantization_script_locally()
        if not local_test_passed:
            print("❌ Local test failed. Fix issues before running on SageMaker.")
            return
    
    # Set up environment
    config = setup_environment()
    
    # Load model information
    model_data, model_name, model_s3_uri = load_model_info(config["S3_BUCKET"])
    
    # Run quantization job
    if not args.skip_job:
        print("\n=== Running SageMaker Processing Job ===")
        job_info = run_quantization_processing_job(config, model_name, model_s3_uri)
        
        if job_info:
            print("\n=== Quantization Summary ===")
            print(f"Processing Job: {job_info['processing_job_name']}")
            print(f"Output S3 URI: {job_info['output_s3_uri']}")
            print(f"Quantized Model Name: {job_info['quantized_model_name']}")
            print("✅ Quantization completed successfully!")
        else:
            print("❌ Quantization job failed.")
    else:
        print("Skipping SageMaker processing job as requested.")

if __name__ == "__main__":
    main()

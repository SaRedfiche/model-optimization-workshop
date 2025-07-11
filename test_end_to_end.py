#!/usr/bin/env python
"""
End-to-end test script for the model optimization workshop.
This script tests the complete workflow from quantization to deployment to cleanup.
"""

import os
import json
import time
import tempfile
import shutil
import tarfile
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.huggingface import HuggingFaceModel

def setup_environment():
    """Set up the environment and return configuration."""
    try:
        sagemaker_session = sagemaker.Session()
        S3_BUCKET = sagemaker_session.default_bucket()
        SAGEMAKER_ROLE_ARN = sagemaker.get_execution_role()
        AWS_REGION = boto3.session.Session().region_name
        OPTIMIZATION_INSTANCE_TYPE = "ml.c5.xlarge"
        
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return None
    
    print(f"Environment setup:")
    print(f"S3_BUCKET: {S3_BUCKET}")
    print(f"AWS_REGION: {AWS_REGION}")
    print(f"OPTIMIZATION_INSTANCE_TYPE: {OPTIMIZATION_INSTANCE_TYPE}")
    
    return {
        "S3_BUCKET": S3_BUCKET,
        "SAGEMAKER_ROLE_ARN": SAGEMAKER_ROLE_ARN,
        "AWS_REGION": AWS_REGION,
        "OPTIMIZATION_INSTANCE_TYPE": OPTIMIZATION_INSTANCE_TYPE,
        "sagemaker_session": sagemaker_session
    }

def wait_for_processing_job(job_name, sagemaker_session, timeout_minutes=15):
    """Wait for processing job to complete."""
    print(f"\n=== Waiting for Processing Job: {job_name} ===")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        try:
            job_description = sagemaker_session.sagemaker_client.describe_processing_job(
                ProcessingJobName=job_name
            )
            job_status = job_description['ProcessingJobStatus']
            
            elapsed_time = time.time() - start_time
            print(f"Status: {job_status} (elapsed: {elapsed_time/60:.1f} min)")
            
            if job_status in ['Completed', 'Failed', 'Stopped']:
                if job_status == 'Completed':
                    print("✅ Processing job completed successfully!")
                    return job_description
                else:
                    print(f"❌ Processing job failed with status: {job_status}")
                    if 'FailureReason' in job_description:
                        print(f"Failure reason: {job_description['FailureReason']}")
                    return None
            
            if elapsed_time > timeout_seconds:
                print(f"❌ Processing job timed out after {timeout_minutes} minutes")
                return None
            
            time.sleep(30)  # Wait 30 seconds before checking again
            
        except Exception as e:
            print(f"Error checking job status: {e}")
            return None

def prepare_model_for_deployment(output_s3_uri, quantized_model_name, S3_BUCKET):
    """Prepare the quantized model for deployment."""
    print(f"\n=== Preparing Model for Deployment ===")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            # Download the quantized model from S3
            print(f"Downloading quantized model from {output_s3_uri}...")
            os.system(f"aws s3 cp --recursive {output_s3_uri} {tmpdirname}/download/")
            
            # Check the structure and move files to the right place
            print(f"Organizing model files...")
            
            # The quantized model is in a subdirectory called 'quantized_model'
            quantized_model_dir = f"{tmpdirname}/download/quantized_model"
            if os.path.exists(quantized_model_dir):
                # Move all files from quantized_model subdirectory to root
                for item in os.listdir(quantized_model_dir):
                    shutil.move(
                        os.path.join(quantized_model_dir, item),
                        os.path.join(tmpdirname, item)
                    )
            else:
                # If no subdirectory, move files from download to root
                for item in os.listdir(f"{tmpdirname}/download"):
                    shutil.move(
                        os.path.join(f"{tmpdirname}/download", item),
                        os.path.join(tmpdirname, item)
                    )
            
            # Create code directory for inference script
            os.makedirs(f"{tmpdirname}/code", exist_ok=True)
            
            # Copy the inference script to the model directory
            print(f"Adding custom inference script...")
            shutil.copy('scripts/inference.py', f"{tmpdirname}/code/inference.py")
            
            # Copy the requirements file for the inference script
            print(f"Adding inference requirements...")
            shutil.copy('scripts/inference_requirements.txt', f"{tmpdirname}/code/requirements.txt")
            
            # List files to verify structure
            print(f"Model directory contents:")
            for root, dirs, files in os.walk(tmpdirname):
                level = root.replace(tmpdirname, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
            
            # Create a tar.gz file
            print(f"Creating model.tar.gz...")
            with tarfile.open(f"{tmpdirname}/model.tar.gz", "w:gz") as tar:
                # Add all files in the temp directory (except the tar file itself)
                for item in os.listdir(tmpdirname):
                    if item != "model.tar.gz" and item != "download":
                        tar.add(os.path.join(tmpdirname, item), arcname=item)
            
            # Upload the tar.gz file to S3
            s3_quantized_model_prefix = f"models/{quantized_model_name}"
            s3_quantized_model_uri = f"s3://{S3_BUCKET}/{s3_quantized_model_prefix}"
            print(f"Uploading model.tar.gz to {s3_quantized_model_uri}...")
            os.system(f"aws s3 cp {tmpdirname}/model.tar.gz {s3_quantized_model_uri}/model.tar.gz")
            
            print(f"✅ Model prepared and uploaded to {s3_quantized_model_uri}/model.tar.gz")
            return f"{s3_quantized_model_uri}/model.tar.gz"
            
        except Exception as e:
            print(f"❌ Error preparing model: {e}")
            return None

def deploy_model(model_data_uri, SAGEMAKER_ROLE_ARN):
    """Deploy the quantized model to a SageMaker endpoint."""
    print(f"\n=== Deploying Model to Endpoint ===")
    
    try:
        # Create HuggingFace model with supported versions
        huggingface_model = HuggingFaceModel(
            model_data=model_data_uri,
            role=SAGEMAKER_ROLE_ARN,
            transformers_version="4.49.0",
            pytorch_version="2.6.0",
            py_version="py312",
            entry_point="inference.py",
            env={
                'HF_TASK': 'text-classification'
            }
        )
        
        # Create a shorter endpoint name
        short_name = "distilbert-quantized-test"
        endpoint_name = f"{short_name}-{int(time.time())}"[-63:]  # Ensure it's under 63 chars
        
        print(f"Deploying to endpoint: {endpoint_name}")
        print("This will take 5-10 minutes...")
        
        # Deploy model to endpoint
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge",
            endpoint_name=endpoint_name
        )
        
        print(f"✅ Model deployed successfully to endpoint: {endpoint_name}")
        return predictor, endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying model: {e}")
        return None, None

def test_inference(predictor):
    """Test inference on the deployed model."""
    print(f"\n=== Testing Inference ===")
    
    test_cases = [
        "I really enjoyed this movie. The acting was great and the plot was engaging.",
        "This movie was terrible. I couldn't even finish watching it.",
        "The weather is nice today.",
        "I love using this product. It works perfectly for my needs.",
        "This service is disappointing and doesn't work as advertised."
    ]
    
    try:
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest {i}: {text}")
            
            start_time = time.time()
            response = predictor.predict({
                "inputs": text
            })
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000
            print(f"Inference time: {inference_time:.2f} ms")
            print(f"Prediction: {response}")
        
        print(f"\n✅ All inference tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during inference testing: {e}")
        return False

def cleanup_resources(endpoint_name, sagemaker_session):
    """Clean up the deployed endpoint."""
    print(f"\n=== Cleaning Up Resources ===")
    
    try:
        if endpoint_name:
            print(f"Deleting endpoint: {endpoint_name}")
            sagemaker_session.delete_endpoint(endpoint_name)
            print(f"✅ Endpoint {endpoint_name} deleted successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False

def main():
    """Run the complete end-to-end test."""
    print("🧪 Model Optimization Workshop - End-to-End Test")
    print("=" * 60)
    
    # Setup environment
    config = setup_environment()
    if not config:
        print("❌ Failed to setup environment")
        return 1
    
    # Check if there's an existing processing job to wait for
    processing_job_name = "quantize-distilbert-base-uncased-2025-07-10-23-49-41-321"
    
    # Wait for processing job to complete
    job_description = wait_for_processing_job(
        processing_job_name, 
        config["sagemaker_session"]
    )
    
    if not job_description:
        print("❌ Processing job failed or timed out")
        return 1
    
    # Get output S3 URI
    output_s3_uri = job_description['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
    quantized_model_name = "distilbert-base-uncased-quantized"
    
    # Prepare model for deployment
    model_data_uri = prepare_model_for_deployment(
        output_s3_uri, 
        quantized_model_name, 
        config["S3_BUCKET"]
    )
    
    if not model_data_uri:
        print("❌ Failed to prepare model for deployment")
        return 1
    
    # Deploy model
    predictor, endpoint_name = deploy_model(
        model_data_uri, 
        config["SAGEMAKER_ROLE_ARN"]
    )
    
    if not predictor:
        print("❌ Failed to deploy model")
        return 1
    
    # Test inference
    inference_success = test_inference(predictor)
    
    # Cleanup resources
    cleanup_success = cleanup_resources(endpoint_name, config["sagemaker_session"])
    
    # Final summary
    print("\n" + "=" * 60)
    if inference_success and cleanup_success:
        print("🎉 End-to-end test PASSED!")
        print("✅ Quantization, deployment, inference, and cleanup all successful!")
        return 0
    else:
        print("❌ End-to-end test FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())

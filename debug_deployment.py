#!/usr/bin/env python
"""
Debug script to check what's causing the py311 error in SageMaker deployment.
"""

import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def debug_deployment():
    """Debug the HuggingFaceModel deployment configuration."""
    
    print("=== Debugging SageMaker HuggingFace Model Deployment ===")
    
    # Get basic info
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    print(f"SageMaker SDK Version: {sagemaker.__version__}")
    print(f"Role: {role}")
    
    # Test HuggingFaceModel configuration
    try:
        print("\n=== Testing HuggingFaceModel Configuration ===")
        
        # This is the exact configuration from the notebook
        huggingface_model = HuggingFaceModel(
            model_data="s3://test-bucket/test-model.tar.gz",  # Dummy S3 path
            role=role,
            transformers_version="4.49.0",
            pytorch_version="2.6.0",
            py_version="py312",
            entry_point="inference.py",
            env={
                'HF_TASK': 'text-classification'
            }
        )
        
        print("✅ HuggingFaceModel created successfully with py312")
        print(f"   - transformers_version: {huggingface_model.transformers_version}")
        print(f"   - pytorch_version: {huggingface_model.pytorch_version}")
        print(f"   - py_version: {huggingface_model.py_version}")
        
    except Exception as e:
        print(f"❌ Error creating HuggingFaceModel: {e}")
        
        # Try with different versions to see what works
        print("\n=== Testing Alternative Configurations ===")
        
        # Test with older versions that might work
        test_configs = [
            {"transformers_version": "4.49.0", "pytorch_version": "2.6.0", "py_version": "py312"},
            {"transformers_version": "4.44.0", "pytorch_version": "2.4.0", "py_version": "py312"},
            {"transformers_version": "4.26.0", "pytorch_version": "1.13.1", "py_version": "py312"},
        ]
        
        for i, config in enumerate(test_configs, 1):
            try:
                test_model = HuggingFaceModel(
                    model_data="s3://test-bucket/test-model.tar.gz",
                    role=role,
                    **config
                )
                print(f"✅ Config {i} works: {config}")
                break
            except Exception as config_error:
                print(f"❌ Config {i} failed: {config} - {config_error}")
    
    # Check available versions
    print("\n=== Checking Available Framework Versions ===")
    try:
        from sagemaker.image_uris import retrieve
        
        # Try to get available versions for HuggingFace
        print("Checking available HuggingFace versions...")
        
        # This might help identify what versions are actually supported
        regions = ['us-east-1', 'us-west-2']
        for region in regions:
            try:
                uri = retrieve(
                    framework='huggingface',
                    region=region,
                    version='4.49.0',
                    py_version='py312',
                    instance_type='ml.g4dn.xlarge',
                    accelerator_type=None,
                    image_scope='inference'
                )
                print(f"✅ HuggingFace 4.49.0 + py312 available in {region}")
                break
            except Exception as uri_error:
                print(f"❌ HuggingFace 4.49.0 + py312 not available in {region}: {uri_error}")
                
    except Exception as version_error:
        print(f"Could not check available versions: {version_error}")

if __name__ == "__main__":
    debug_deployment()

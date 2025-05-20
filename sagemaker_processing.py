"""
Utility functions for running optimization tasks on separate SageMaker instances.
This allows using a small notebook instance while offloading heavy computation to larger instances.
"""

import os
import json
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.pytorch.processing import PyTorchProcessor

def run_optimization_job(
    script_path,
    instance_type="ml.g4dn.xlarge",
    inputs=None,
    outputs=None,
    job_name=None,
    role=None,
    framework_version="1.13.1",
    py_version="py39",
    wait=True
):
    """
    Run an optimization task on a separate SageMaker instance.
    
    Args:
        script_path: Path to the Python script that performs the optimization
        instance_type: SageMaker instance type to use
        inputs: Dictionary of input data sources
        outputs: Dictionary of output destinations
        job_name: Name for the processing job
        role: SageMaker execution role ARN
        framework_version: PyTorch version
        py_version: Python version
        wait: Whether to wait for the job to complete
        
    Returns:
        SageMaker processing job object
    """
    # Get SageMaker session and role
    sagemaker_session = sagemaker.Session()
    if role is None:
        role = sagemaker.get_execution_role()
    
    # Generate job name if not provided
    if job_name is None:
        import time
        job_name = f"model-optimization-{int(time.time())}"
    
    # Create processor
    processor = PyTorchProcessor(
        framework_version=framework_version,
        py_version=py_version,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name=job_name,
        sagemaker_session=sagemaker_session
    )
    
    # Prepare inputs
    processing_inputs = []
    if inputs:
        for name, source in inputs.items():
            processing_inputs.append(
                ProcessingInput(
                    source=source,
                    destination=f"/opt/ml/processing/input/{name}",
                    input_name=name
                )
            )
    
    # Prepare outputs
    processing_outputs = []
    if outputs:
        for name, destination in outputs.items():
            processing_outputs.append(
                ProcessingOutput(
                    source=f"/opt/ml/processing/output/{name}",
                    destination=destination,
                    output_name=name
                )
            )
    
    # Run the processing job
    processor.run(
        code=script_path,
        inputs=processing_inputs,
        outputs=processing_outputs,
        wait=wait
    )
    
    return processor

def run_quantization_job(
    model_key,
    model_info,
    s3_bucket,
    instance_type="ml.c5.xlarge",
    role=None
):
    """
    Run quantization on a separate SageMaker instance.
    
    Args:
        model_key: Key of the model to quantize
        model_info: Dictionary with model information
        s3_bucket: S3 bucket for storing inputs and outputs
        instance_type: SageMaker instance type to use
        role: SageMaker execution role ARN
        
    Returns:
        S3 path to the quantized model
    """
    # Save model info to a temporary file
    with open('temp_model_info.json', 'w') as f:
        json.dump({model_key: model_info[model_key]}, f)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        'temp_model_info.json', 
        s3_bucket, 
        f'optimization/inputs/{model_key}/model_info.json'
    )
    
    # Define inputs and outputs
    inputs = {
        'model_info': f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json'
    }
    
    outputs = {
        'quantized_model': f's3://{s3_bucket}/optimization/outputs/{model_key}/quantized'
    }
    
    # Run the job
    processor = run_optimization_job(
        script_path='quantization_script.py',
        instance_type=instance_type,
        inputs=inputs,
        outputs=outputs,
        job_name=f'quantize-{model_key}',
        role=role
    )
    
    # Return the S3 path to the quantized model
    return f's3://{s3_bucket}/optimization/outputs/{model_key}/quantized'

def run_pruning_job(
    model_key,
    model_info,
    s3_bucket,
    instance_type="ml.g4dn.xlarge",
    role=None
):
    """
    Run pruning on a separate SageMaker instance.
    
    Args:
        model_key: Key of the model to prune
        model_info: Dictionary with model information
        s3_bucket: S3 bucket for storing inputs and outputs
        instance_type: SageMaker instance type to use
        role: SageMaker execution role ARN
        
    Returns:
        S3 path to the pruned model
    """
    # Save model info to a temporary file
    with open('temp_model_info.json', 'w') as f:
        json.dump({model_key: model_info[model_key]}, f)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        'temp_model_info.json', 
        s3_bucket, 
        f'optimization/inputs/{model_key}/model_info.json'
    )
    
    # Define inputs and outputs
    inputs = {
        'model_info': f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json'
    }
    
    outputs = {
        'pruned_model': f's3://{s3_bucket}/optimization/outputs/{model_key}/pruned'
    }
    
    # Run the job
    processor = run_optimization_job(
        script_path='pruning_script.py',
        instance_type=instance_type,
        inputs=inputs,
        outputs=outputs,
        job_name=f'prune-{model_key}',
        role=role
    )
    
    # Return the S3 path to the pruned model
    return f's3://{s3_bucket}/optimization/outputs/{model_key}/pruned'

def run_distillation_job(
    model_key,
    model_info,
    s3_bucket,
    instance_type="ml.g4dn.2xlarge",
    role=None
):
    """
    Run knowledge distillation on a separate SageMaker instance.
    
    Args:
        model_key: Key of the model to distill
        model_info: Dictionary with model information
        s3_bucket: S3 bucket for storing inputs and outputs
        instance_type: SageMaker instance type to use
        role: SageMaker execution role ARN
        
    Returns:
        S3 path to the distilled model
    """
    # Save model info to a temporary file
    with open('temp_model_info.json', 'w') as f:
        json.dump({model_key: model_info[model_key]}, f)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        'temp_model_info.json', 
        s3_bucket, 
        f'optimization/inputs/{model_key}/model_info.json'
    )
    
    # Define inputs and outputs
    inputs = {
        'model_info': f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json'
    }
    
    outputs = {
        'distilled_model': f's3://{s3_bucket}/optimization/outputs/{model_key}/distilled'
    }
    
    # Run the job
    processor = run_optimization_job(
        script_path='distillation_script.py',
        instance_type=instance_type,
        inputs=inputs,
        outputs=outputs,
        job_name=f'distill-{model_key}',
        role=role
    )
    
    # Return the S3 path to the distilled model
    return f's3://{s3_bucket}/optimization/outputs/{model_key}/distilled'

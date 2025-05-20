"""
Utility functions for running distributed processing jobs on SageMaker.
"""

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.pytorch.processing import PyTorchProcessor

def run_quantization_job(processor, model_key, model_info, s3_bucket, 
                         quantization_method='dynamic', quantization_bits=8):
    """
    Run a quantization job on SageMaker Processing.
    
    Args:
        processor: SageMaker PyTorchProcessor instance
        model_key: Key identifying the model
        model_info: Dictionary with model information
        s3_bucket: S3 bucket name
        quantization_method: Method for quantization ('dynamic', 'static', or 'aware')
        quantization_bits: Number of bits for quantization (8 or 16)
        
    Returns:
        SageMaker processing job
    """
    # Define inputs and outputs
    inputs = [
        ProcessingInput(
            source=f's3://{s3_bucket}/scripts/quantization_script.py',
            destination='/opt/ml/processing/input/code/quantization_script.py'
        ),
        ProcessingInput(
            source=f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json',
            destination='/opt/ml/processing/input/data/model_info.json'
        )
    ]
    
    outputs = [
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{s3_bucket}/optimization/outputs/{model_key}'
        )
    ]
    
    # Run the processing job
    job = processor.run(
        code='quantization_script.py',
        inputs=inputs,
        outputs=outputs,
        arguments=[
            '--model-info-path', '/opt/ml/processing/input/data/model_info.json',
            '--output-dir', '/opt/ml/processing/output',
            '--quantization-method', quantization_method,
            '--quantization-bits', str(quantization_bits)
        ]
    )
    
    return job

def run_pruning_job(processor, model_key, model_info, s3_bucket, 
                    pruning_method='l1_unstructured', pruning_amount=0.3):
    """
    Run a pruning job on SageMaker Processing.
    
    Args:
        processor: SageMaker PyTorchProcessor instance
        model_key: Key identifying the model
        model_info: Dictionary with model information
        s3_bucket: S3 bucket name
        pruning_method: Method for pruning ('l1_unstructured', 'random_unstructured', or 'ln_structured')
        pruning_amount: Amount of weights to prune (0.0 to 1.0)
        
    Returns:
        SageMaker processing job
    """
    # Define inputs and outputs
    inputs = [
        ProcessingInput(
            source=f's3://{s3_bucket}/scripts/pruning_script.py',
            destination='/opt/ml/processing/input/code/pruning_script.py'
        ),
        ProcessingInput(
            source=f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json',
            destination='/opt/ml/processing/input/data/model_info.json'
        )
    ]
    
    outputs = [
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{s3_bucket}/optimization/outputs/{model_key}'
        )
    ]
    
    # Run the processing job
    job = processor.run(
        code='pruning_script.py',
        inputs=inputs,
        outputs=outputs,
        arguments=[
            '--model-info-path', '/opt/ml/processing/input/data/model_info.json',
            '--output-dir', '/opt/ml/processing/output',
            '--pruning-method', pruning_method,
            '--pruning-amount', str(pruning_amount)
        ]
    )
    
    return job

def run_distillation_job(processor, model_key, teacher_model_info, s3_bucket, 
                         student_model_name='distilbert-base-uncased', 
                         num_epochs=3, batch_size=8):
    """
    Run a knowledge distillation job on SageMaker Processing.
    
    Args:
        processor: SageMaker PyTorchProcessor instance
        model_key: Key identifying the model
        teacher_model_info: Dictionary with teacher model information
        s3_bucket: S3 bucket name
        student_model_name: Name of the student model architecture
        num_epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        SageMaker processing job
    """
    # Define inputs and outputs
    inputs = [
        ProcessingInput(
            source=f's3://{s3_bucket}/scripts/distillation_script.py',
            destination='/opt/ml/processing/input/code/distillation_script.py'
        ),
        ProcessingInput(
            source=f's3://{s3_bucket}/optimization/inputs/{model_key}/model_info.json',
            destination='/opt/ml/processing/input/data/model_info.json'
        )
    ]
    
    outputs = [
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{s3_bucket}/optimization/outputs/{model_key}_distilled'
        )
    ]
    
    # Run the processing job
    job = processor.run(
        code='distillation_script.py',
        inputs=inputs,
        outputs=outputs,
        arguments=[
            '--model-info-path', '/opt/ml/processing/input/data/model_info.json',
            '--output-dir', '/opt/ml/processing/output',
            '--student-model-name', student_model_name,
            '--num-epochs', str(num_epochs),
            '--batch-size', str(batch_size)
        ]
    )
    
    return job

#!/usr/bin/env python
"""
Model Optimization Workshop: Knowledge Distillation

This script performs the same operations as the 04_knowledge_distillation.ipynb notebook.
It launches SageMaker Processing jobs to perform knowledge distillation on larger instances.
"""

import os
import json
import time
import argparse
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor

def setup_environment():
    """Set up the environment and return configuration."""
    # Load workshop configuration
    try:
        with open('workshop_config.json', 'r') as f:
            workshop_config = json.load(f)
        print("✅ Workshop configuration loaded successfully")
    except FileNotFoundError:
        print("❌ workshop_config.json not found. Please run notebook 1 first.")
        return None
    
    return workshop_config

def load_model_info():
    """Load model information from previous notebooks."""
    try:
        with open('model_info.json', 'r') as f:
            model_info_dict = json.load(f)
        print(f"✅ Loaded model information for {len(model_info_dict)} models")
        return model_info_dict
    except FileNotFoundError:
        print("❌ model_info.json not found. Please run previous notebooks first.")
        return None

def create_sagemaker_processor(workshop_config):
    """Create SageMaker PyTorch processor for distillation jobs."""
    try:
        # Set up SageMaker session
        sagemaker_session = sagemaker.Session()
        role = workshop_config['role']
        
        # Create PyTorch processor for distillation
        processor = PyTorchProcessor(
            framework_version='2.6.0',
            py_version='py312',
            role=role,
            instance_type='ml.m5.xlarge',  # Use appropriate instance type
            instance_count=1,
            base_job_name='knowledge-distillation',
            sagemaker_session=sagemaker_session
        )
        
        print(f"✅ Created PyTorch processor")
        return processor, sagemaker_session
        
    except Exception as e:
        print(f"❌ Error creating SageMaker processor: {e}")
        return None, None

def define_student_architectures():
    """Define student model architectures."""
    student_architectures = {
        "distilbert-small": {
            "model_name": "distilbert-base-uncased",
            "num_hidden_layers": 3,
            "hidden_size": 384,
            "description": "Small DistilBERT with 3 layers and 384 hidden size"
        },
        "distilbert-tiny": {
            "model_name": "distilbert-base-uncased",
            "num_hidden_layers": 2,
            "hidden_size": 256,
            "description": "Tiny DistilBERT with 2 layers and 256 hidden size"
        }
    }
    
    print("Available student architectures:")
    for arch_name, config in student_architectures.items():
        print(f"  {arch_name}: {config['description']}")
    
    return student_architectures

def create_and_upload_config_files(model_info_dict, student_architectures, workshop_config):
    """Create and upload teacher and student configuration files to S3."""
    s3_client = boto3.client('s3')
    bucket = workshop_config['s3_bucket']
    prefix = workshop_config['s3_prefix']
    
    config_files = {}
    
    # Filter models suitable for distillation (text classification models)
    distillable_models = {k: v for k, v in model_info_dict.items() 
                         if v.get('task') == 'text-classification'}
    
    print(f"Found {len(distillable_models)} models suitable for knowledge distillation")
    
    # Create teacher config files
    for model_key in distillable_models.keys():
        model_info = distillable_models[model_key]
        
        # Create teacher info file
        teacher_info = {model_key: model_info}
        teacher_filename = f"teacher_{model_key}_info.json"
        
        with open(teacher_filename, 'w') as f:
            json.dump(teacher_info, f, indent=2)
        
        # Upload teacher info to S3
        teacher_s3_key = f"{prefix}/distillation/teacher_configs/{teacher_filename}"
        s3_client.upload_file(teacher_filename, bucket, teacher_s3_key)
        teacher_s3_uri = f"s3://{bucket}/{teacher_s3_key}"
        
        config_files[f"teacher_{model_key}"] = teacher_s3_uri
        print(f"✅ Uploaded teacher config: {teacher_s3_uri}")
    
    # Create student config files
    for arch_name, config in student_architectures.items():
        student_info = {arch_name: config}
        student_filename = f"student_{arch_name}_info.json"
        
        with open(student_filename, 'w') as f:
            json.dump(student_info, f, indent=2)
        
        # Upload student info to S3
        student_s3_key = f"{prefix}/distillation/student_configs/{student_filename}"
        s3_client.upload_file(student_filename, bucket, student_s3_key)
        student_s3_uri = f"s3://{bucket}/{student_s3_key}"
        
        config_files[f"student_{arch_name}"] = student_s3_uri
        print(f"✅ Uploaded student config: {student_s3_uri}")
    
    return config_files, distillable_models

def launch_distillation_jobs(processor, config_files, distillable_models, student_architectures, workshop_config, args):
    """Launch knowledge distillation processing jobs."""
    bucket = workshop_config['s3_bucket']
    job_names = []
    job_output_paths = {}
    
    print("\n🚀 Launching distillation jobs...")
    
    for model_key in distillable_models.keys():
        teacher_s3_uri = config_files[f"teacher_{model_key}"]
        teacher_filename = f"teacher_{model_key}_info.json"
        
        for arch_name in student_architectures.keys():
            student_s3_uri = config_files[f"student_{arch_name}"]
            student_filename = f"student_{arch_name}_info.json"
            
            # Define job key and output path
            job_key = f"{model_key}-{arch_name}"
            output_path = f"s3://{bucket}/optimization/outputs/{job_key}-distilled"
            job_output_paths[job_key] = output_path
            
            # Define inputs and outputs
            inputs = [
                ProcessingInput(
                    source=teacher_s3_uri,
                    destination='/opt/ml/processing/input/teacher'
                ),
                ProcessingInput(
                    source=student_s3_uri,
                    destination='/opt/ml/processing/input/student'
                )
            ]
            
            outputs = [
                ProcessingOutput(
                    output_name='distilled-model',
                    source='/opt/ml/processing/output',
                    destination=output_path
                )
            ]
            
            # Create job arguments
            arguments = [
                '--teacher-info-path', f'/opt/ml/processing/input/teacher/{teacher_filename}',
                '--student-info-path', f'/opt/ml/processing/input/student/{student_filename}',
                '--temperature', str(args.temperature),
                '--alpha', str(args.alpha),
                '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size)
            ]
            
            try:
                # Create unique job name
                timestamp = int(time.time())
                job_name = f"distillation-{job_key}-{timestamp}"
                
                # Launch processing job
                processor.run(
                    code='distillation_script.py',
                    source_dir='distillation_scripts',
                    inputs=inputs,
                    outputs=outputs,
                    arguments=arguments,
                    wait=False,
                    job_name=job_name
                )
                
                job_names.append(job_name)
                print(f"✅ Launched job for {job_key}: {job_name}")
                
            except Exception as e:
                print(f"❌ Error launching job for {job_key}: {e}")
    
    return job_names, job_output_paths

def monitor_jobs(job_names):
    """Monitor the status of distillation jobs."""
    if not job_names:
        print("No jobs to monitor.")
        return
    
    print(f"\n📊 Monitoring {len(job_names)} distillation jobs...")
    
    sagemaker_client = boto3.client('sagemaker')
    
    for job_name in job_names:
        try:
            response = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
            status = response['ProcessingJobStatus']
            print(f"  {job_name}: {status}")
            
            if status == 'Failed':
                failure_reason = response.get('FailureReason', 'No failure reason provided')
                print(f"    Failure reason: {failure_reason}")
                
        except Exception as e:
            print(f"  Error checking {job_name}: {e}")

def analyze_results(job_output_paths, workshop_config):
    """Analyze distillation results from S3."""
    print(f"\n📈 Analyzing distillation results...")
    
    s3_client = boto3.client('s3')
    bucket = workshop_config['s3_bucket']
    results = []
    
    for job_key, output_path in job_output_paths.items():
        try:
            # Check if distillation info file exists
            info_key = output_path.replace(f's3://{bucket}/', '') + '/distillation_info.json'
            
            try:
                response = s3_client.get_object(Bucket=bucket, Key=info_key)
                info_content = response['Body'].read().decode('utf-8')
                info_data = json.loads(info_content)
                
                results.append({
                    'job_key': job_key,
                    'teacher_model': info_data['teacher_model'],
                    'student_config': info_data['student_config'],
                    'teacher_parameters': info_data['teacher_parameters'],
                    'student_parameters': info_data['student_parameters'],
                    'size_reduction_percent': info_data['size_reduction_percent'],
                    'temperature': info_data['temperature'],
                    'alpha': info_data['alpha'],
                    'epochs': info_data['epochs'],
                    'output_path': output_path
                })
                
                print(f"✅ Distilled model found at {output_path}")
                print(f"   Teacher parameters: {info_data['teacher_parameters']:,}")
                print(f"   Student parameters: {info_data['student_parameters']:,}")
                print(f"   Size reduction: {info_data['size_reduction_percent']:.2f}%")
                
            except s3_client.exceptions.NoSuchKey:
                print(f"⚠️ No distilled model found at {output_path}")
                print(f"   The distillation job for {job_key} may have failed.")
                
        except Exception as e:
            print(f"❌ Error checking results for {job_key}: {e}")
    
    if results:
        print(f"\n{'='*60}")
        print("KNOWLEDGE DISTILLATION SUMMARY")
        print(f"{'='*60}")
        
        for result in results:
            print(f"Job: {result['job_key']}")
            print(f"  Teacher: {result['teacher_model']}")
            print(f"  Student: {result['student_config']}")
            print(f"  Parameters: {result['teacher_parameters']:,} → {result['student_parameters']:,}")
            print(f"  Size Reduction: {result['size_reduction_percent']:.2f}%")
            print(f"  Training: T={result['temperature']}, α={result['alpha']}, epochs={result['epochs']}")
            print()
    else:
        print("No successful distillation results found.")
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform knowledge distillation using SageMaker Processing")
    parser.add_argument("--temperature", type=float, default=2.0, 
                        help="Temperature for softening teacher outputs")
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Weight for distillation loss vs. task loss")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--monitor", action="store_true", 
                        help="Monitor existing jobs instead of launching new ones")
    parser.add_argument("--analyze", action="store_true", 
                        help="Analyze results from previous jobs")
    return parser.parse_args()

def main():
    """Main function to run the knowledge distillation orchestration."""
    args = parse_args()
    
    print("🧠 Knowledge Distillation with SageMaker Processing")
    print("=" * 60)
    
    # Set up environment
    workshop_config = setup_environment()
    if not workshop_config:
        return
    
    # Load model information
    model_info_dict = load_model_info()
    if not model_info_dict:
        return
    
    # Define student architectures
    student_architectures = define_student_architectures()
    
    if args.analyze:
        # Just analyze existing results
        print("📊 Analyzing existing results...")
        # This would need job output paths from previous runs
        # For now, just show the pattern
        print("To analyze results, you need the job output paths from previous runs.")
        return
    
    if args.monitor:
        # Just monitor existing jobs
        print("📊 Monitoring existing jobs...")
        # This would need job names from previous runs
        # For now, just show the pattern
        print("To monitor jobs, you need the job names from previous runs.")
        return
    
    # Create SageMaker processor
    processor, sagemaker_session = create_sagemaker_processor(workshop_config)
    if not processor:
        return
    
    # Create and upload configuration files
    config_files, distillable_models = create_and_upload_config_files(
        model_info_dict, student_architectures, workshop_config
    )
    
    if not distillable_models:
        print("❌ No suitable models found for distillation")
        return
    
    # Launch distillation jobs
    job_names, job_output_paths = launch_distillation_jobs(
        processor, config_files, distillable_models, student_architectures, 
        workshop_config, args
    )
    
    if job_names:
        print(f"\n✅ Successfully launched {len(job_names)} distillation jobs")
        print("You can monitor their progress in the SageMaker console.")
        
        # Save job information for later monitoring/analysis
        job_info = {
            'job_names': job_names,
            'job_output_paths': job_output_paths,
            'timestamp': time.time()
        }
        
        with open('distillation_jobs.json', 'w') as f:
            json.dump(job_info, f, indent=2)
        
        print("Job information saved to distillation_jobs.json")
        
        # Optional: Monitor jobs briefly
        time.sleep(10)  # Wait a bit for jobs to start
        monitor_jobs(job_names)
        
    else:
        print("❌ No jobs were launched successfully")

if __name__ == "__main__":
    main()

"""
Utility functions for model optimization workshop.
This file contains helper functions for robustness and observability.
"""

import json
import time
import boto3
import traceback

def analyze_job_failure(job_name):
    """Analyze a failed SageMaker job and provide diagnostic information."""
    client = boto3.client('sagemaker')
    response = client.describe_processing_job(ProcessingJobName=job_name)
    
    failure_reason = response.get('FailureReason', 'No failure reason provided')
    print(f"Job failed with reason: {failure_reason}")
    
    # Get CloudWatch logs URL
    log_group = f"/aws/sagemaker/ProcessingJobs"
    log_stream = f"{job_name}/algo-1"
    region = boto3.session.Session().region_name
    logs_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#logStream:group={log_group};streamFilter=typeLogStreamPrefix;prefix={log_stream}"
    
    print(f"View detailed logs at: {logs_url}")
    return failure_reason

def save_checkpoint(s3_bucket, job_name, stage, data):
    """Save a checkpoint to S3 to track progress."""
    s3_client = boto3.client('s3')
    checkpoint_key = f"checkpoints/{job_name}/{stage}.json"
    
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=checkpoint_key,
        Body=json.dumps(data)
    )
    
    print(f"Saved checkpoint for stage '{stage}' to s3://{s3_bucket}/{checkpoint_key}")

def handle_processing_error(e, job_name=None):
    """Handle processing errors consistently across notebooks."""
    print(f"Error: {str(e)}")
    print("Stack trace:")
    traceback.print_exc()
    
    if job_name:
        print(f"\nAnalyzing job {job_name} for more details:")
        analyze_job_failure(job_name)
    
    print("\nTroubleshooting steps:")
    print("1. Check that all required dependencies are included")
    print("2. Verify S3 paths and permissions")
    print("3. Check for sufficient instance resources")
    print("4. Review CloudWatch logs for detailed error messages")
    print("5. Ensure the script has the correct permissions")
    print("6. Verify that the model is compatible with the optimization technique")
    print("7. Try with a smaller model or larger instance type")

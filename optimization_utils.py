"""
Utility functions for model optimization workshop.
This file contains helper functions for robustness and observability.
"""

import json
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

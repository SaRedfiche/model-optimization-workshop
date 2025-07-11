#!/usr/bin/env python3
"""
Cleanup script to delete the deployed sentiment analysis endpoint.
"""

import boto3
import sagemaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_endpoint(endpoint_name):
    """Delete the specified endpoint"""
    try:
        # Get SageMaker client
        sagemaker_client = boto3.client('sagemaker')
        
        logger.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        logger.info(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        
        logger.info("Cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

if __name__ == "__main__":
    # The endpoint name from the deployment
    endpoint_name = "sentiment-analysis-1752276102"
    
    print(f"Cleaning up endpoint: {endpoint_name}")
    cleanup_endpoint(endpoint_name)
    print("Endpoint cleanup completed!")

#!/usr/bin/env python
import sagemaker
import boto3

print('Testing SageMaker setup...')

try:
    # Test SageMaker session
    session = sagemaker.Session()
    print(f'Region: {session.boto_region_name}')
    
    # Test execution role
    try:
        role = sagemaker.get_execution_role()
        print(f'Execution role: {role}')
    except Exception as e:
        print(f'Execution role error: {e}')
        
        # Try to get caller identity
        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            print(f'AWS Identity: {identity}')
            
            # Suggest a role ARN format
            account_id = identity['Account']
            suggested_role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
            print(f'Suggested role ARN: {suggested_role}')
            
        except Exception as sts_error:
            print(f'STS error: {sts_error}')
    
except Exception as e:
    print(f'SageMaker session error: {e}')

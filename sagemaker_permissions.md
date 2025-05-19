# SageMaker Permissions for Model Optimization Workshop

This document outlines the AWS permissions required to run the Model Optimization Workshop in Amazon SageMaker.

## Required IAM Permissions

The SageMaker execution role used for this workshop needs the following permissions:

### SageMaker Permissions
- `sagemaker:CreateTrainingJob`
- `sagemaker:CreateModel`
- `sagemaker:CreateEndpoint`
- `sagemaker:CreateEndpointConfig`
- `sagemaker:InvokeEndpoint`
- `sagemaker:DeleteEndpoint`
- `sagemaker:DeleteEndpointConfig`
- `sagemaker:DeleteModel`

### S3 Permissions
- `s3:CreateBucket`
- `s3:ListBucket`
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`

### CloudWatch Permissions
- `logs:CreateLogGroup`
- `logs:CreateLogStream`
- `logs:PutLogEvents`
- `logs:DescribeLogStreams`

## Recommended IAM Policy

The simplest approach is to attach the following AWS managed policies to your SageMaker execution role:

1. `AmazonSageMakerFullAccess`
2. `AmazonS3FullAccess`

## Minimal Custom Policy

If you prefer a more restrictive approach, you can create a custom policy with the following JSON:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:InvokeEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::model-optimization-workshop-*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::model-optimization-workshop-*/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "*"
        }
    ]
}
```

## SageMaker Notebook Instance Setup

When creating a SageMaker notebook instance for this workshop:

1. Choose an instance type with sufficient memory and compute (ml.t3.xlarge or larger recommended)
2. Use the standard ML AMI provided by SageMaker
3. Attach an IAM role with the permissions described above
4. Ensure the notebook instance has internet access to download models from Hugging Face

## Execution Role in Notebooks

The workshop notebooks are designed to automatically detect and use the SageMaker execution role. This is done with the following code:

```python
import sagemaker

# Get the SageMaker session
sagemaker_session = sagemaker.Session()

# Get the execution role
role = sagemaker.get_execution_role()
```

This role will be used for all AWS operations throughout the workshop, including creating S3 buckets, deploying models to SageMaker endpoints, and cleaning up resources.

# SageMaker Execution Role Setup

This document describes the IAM role that was created for the Model Optimization Workshop.

## Role Details

- **Role Name**: `ModelOptimizationWorkshopRole`
- **ARN**: `arn:aws:iam::076541649554:role/ModelOptimizationWorkshopRole`
- **Description**: Execution role for Model Optimization Workshop SageMaker notebooks

## Attached Policies

The role has the following policies attached:

1. **AmazonSageMakerFullAccess**
   - Provides full access to SageMaker resources
   - Allows creating, configuring, and managing SageMaker notebook instances, models, endpoints, etc.

2. **AmazonS3FullAccess**
   - Provides full access to S3 resources
   - Allows storing and retrieving models, datasets, and other artifacts

3. **ModelOptimizationWorkshopPolicy** (Custom Policy)
   - ARN: `arn:aws:iam::076541649554:policy/ModelOptimizationWorkshopPolicy`
   - Provides access to CloudWatch metrics and logs
   - Allows the workshop notebooks to create and access logs and metrics

## Trust Relationship

The role trusts the SageMaker service to assume it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## Using the Role

When creating a SageMaker notebook instance for the workshop:

1. Go to the SageMaker console
2. Click on "Notebook instances" and then "Create notebook instance"
3. In the "Permissions and encryption" section, select "Existing role"
4. Choose `ModelOptimizationWorkshopRole` from the dropdown
5. Complete the rest of the notebook instance configuration

This role provides all the necessary permissions for running the Model Optimization Workshop notebooks.

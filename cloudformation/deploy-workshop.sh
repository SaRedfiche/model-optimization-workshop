#!/bin/bash
# Script to deploy the Model Optimization Workshop using CloudFormation

# Default values
STACK_NAME="model-optimization-workshop"
REGION=$(aws configure get region)
NOTEBOOK_INSTANCE_TYPE="ml.t3.medium"
OPTIMIZATION_INSTANCE_TYPE="ml.c5.xlarge"
WORKSHOP_NAME="model-optimization-workshop"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --stack-name)
      STACK_NAME="$2"
      shift
      shift
      ;;
    --region)
      REGION="$2"
      shift
      shift
      ;;
    --notebook-instance)
      NOTEBOOK_INSTANCE_TYPE="$2"
      shift
      shift
      ;;
    --optimization-instance)
      OPTIMIZATION_INSTANCE_TYPE="$2"
      shift
      shift
      ;;
    --workshop-name)
      WORKSHOP_NAME="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --stack-name NAME           CloudFormation stack name (default: model-optimization-workshop)"
      echo "  --region REGION             AWS region (default: from AWS CLI configuration)"
      echo "  --notebook-instance TYPE    SageMaker notebook instance type (default: ml.t3.medium)"
      echo "  --optimization-instance TYPE SageMaker processing instance type (default: ml.c5.xlarge)"
      echo "  --workshop-name NAME        Name for workshop resources (default: model-optimization-workshop)"
      echo "  --help                      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate region
if [ -z "$REGION" ]; then
  echo "Error: AWS region not specified and not found in AWS CLI configuration"
  echo "Please specify a region with --region or configure AWS CLI with 'aws configure'"
  exit 1
fi

echo "Deploying Model Optimization Workshop with the following settings:"
echo "  Stack Name: $STACK_NAME"
echo "  Region: $REGION"
echo "  Notebook Instance Type: $NOTEBOOK_INSTANCE_TYPE"
echo "  Optimization Instance Type: $OPTIMIZATION_INSTANCE_TYPE"
echo "  Workshop Name: $WORKSHOP_NAME"
echo ""

# Deploy CloudFormation stack
echo "Creating CloudFormation stack..."
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-body file://workshop-template.yaml \
  --parameters \
    ParameterKey=NotebookInstanceType,ParameterValue=$NOTEBOOK_INSTANCE_TYPE \
    ParameterKey=OptimizationInstanceType,ParameterValue=$OPTIMIZATION_INSTANCE_TYPE \
    ParameterKey=WorkshopName,ParameterValue=$WORKSHOP_NAME \
  --capabilities CAPABILITY_IAM \
  --region $REGION

# Wait for stack creation to complete
echo "Waiting for stack creation to complete (this may take several minutes)..."
aws cloudformation wait stack-create-complete \
  --stack-name $STACK_NAME \
  --region $REGION

if [ $? -eq 0 ]; then
  echo "Stack creation completed successfully!"
  
  # Get stack outputs
  NOTEBOOK_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='NotebookInstanceUrl'].OutputValue" \
    --output text \
    --region $REGION)
  
  BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='WorkshopBucketName'].OutputValue" \
    --output text \
    --region $REGION)
  
  ROLE_ARN=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='SageMakerRoleArn'].OutputValue" \
    --output text \
    --region $REGION)
  
  echo ""
  echo "Workshop Resources:"
  echo "  Notebook URL: $NOTEBOOK_URL"
  echo "  S3 Bucket: $BUCKET_NAME"
  echo "  SageMaker Role ARN: $ROLE_ARN"
  echo ""
  echo "Note: It may take a few more minutes for the notebook instance to be fully ready."
  echo "You can check the status in the SageMaker console."
else
  echo "Stack creation failed. Check the AWS CloudFormation console for details."
  exit 1
fi

#!/bin/bash
# Script to clean up the Model Optimization Workshop resources

# Default values
STACK_NAME="model-optimization-workshop"
REGION=$(aws configure get region)
EMPTY_BUCKET=false

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
    --empty-bucket)
      EMPTY_BUCKET=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --stack-name NAME    CloudFormation stack name (default: model-optimization-workshop)"
      echo "  --region REGION      AWS region (default: from AWS CLI configuration)"
      echo "  --empty-bucket       Empty the S3 bucket before deleting the stack"
      echo "  --help               Show this help message"
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

echo "Cleaning up Model Optimization Workshop resources:"
echo "  Stack Name: $STACK_NAME"
echo "  Region: $REGION"
echo "  Empty Bucket: $EMPTY_BUCKET"
echo ""

# Get bucket name from stack outputs
BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query "Stacks[0].Outputs[?OutputKey=='WorkshopBucketName'].OutputValue" \
  --output text \
  --region $REGION)

if [ -n "$BUCKET_NAME" ] && [ "$EMPTY_BUCKET" = true ]; then
  echo "Emptying S3 bucket: $BUCKET_NAME"
  aws s3 rm s3://$BUCKET_NAME --recursive --region $REGION
fi

# Delete CloudFormation stack
echo "Deleting CloudFormation stack: $STACK_NAME"
aws cloudformation delete-stack \
  --stack-name $STACK_NAME \
  --region $REGION

# Wait for stack deletion to complete
echo "Waiting for stack deletion to complete (this may take several minutes)..."
aws cloudformation wait stack-delete-complete \
  --stack-name $STACK_NAME \
  --region $REGION

if [ $? -eq 0 ]; then
  echo "Stack deletion completed successfully!"
else
  echo "Stack deletion may have encountered issues. Check the AWS CloudFormation console for details."
  exit 1
fi

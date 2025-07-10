#!/bin/bash
# Script to run the fine-tuning job

# Set your S3 bucket
S3_BUCKET="your-s3-bucket"  # Replace with your S3 bucket name
S3_PREFIX="sentiment-analysis"

# Set training parameters
MODEL_ID="distilbert-base-uncased"
DATASET_NAME="glue"
DATASET_CONFIG="sst2"
INSTANCE_TYPE="ml.g4dn.xlarge"
EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=5e-5

# Run the fine-tuning script
python 05_fine_tuning.py \
    --model_id ${MODEL_ID} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config ${DATASET_CONFIG} \
    --s3_bucket ${S3_BUCKET} \
    --s3_prefix ${S3_PREFIX} \
    --instance_type ${INSTANCE_TYPE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE}

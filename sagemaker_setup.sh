#!/bin/bash
# Script to set up the Model Optimization Workshop in a SageMaker notebook instance

# Download the zip file from S3
aws s3 cp s3://model-optimization-workshop-bucket/model-optimization-workshop.zip .

# Unzip the file
unzip -o model-optimization-workshop.zip -d model-optimization-workshop

# Change to the workshop directory
cd model-optimization-workshop

# Install required packages
pip install -r requirements.txt

echo "Setup complete! You can now navigate to the model-optimization-workshop directory and start with notebook 01_introduction.ipynb"

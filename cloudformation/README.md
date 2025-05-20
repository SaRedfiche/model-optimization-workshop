# Model Optimization Workshop - CloudFormation Deployment

This directory contains CloudFormation templates and scripts to automate the deployment of the Model Optimization Workshop infrastructure.

## What Gets Deployed

The CloudFormation template creates the following resources:

1. **S3 Bucket** - For storing models, scripts, and optimization outputs
2. **IAM Role** - With permissions for SageMaker and S3 access
3. **SageMaker Notebook Instance** - For running the workshop notebooks

## Deployment Instructions

### Prerequisites

- AWS CLI installed and configured with appropriate permissions
- Bash shell environment

### Deploying the Workshop

1. Make the deployment script executable:
   ```bash
   chmod +x deploy-workshop.sh
   ```

2. Run the deployment script:
   ```bash
   ./deploy-workshop.sh
   ```

3. Customize the deployment with options:
   ```bash
   ./deploy-workshop.sh --notebook-instance ml.t3.large --optimization-instance ml.g4dn.xlarge
   ```

### Available Options

- `--stack-name NAME` - CloudFormation stack name (default: model-optimization-workshop)
- `--region REGION` - AWS region (default: from AWS CLI configuration)
- `--notebook-instance TYPE` - SageMaker notebook instance type (default: ml.t3.medium)
- `--optimization-instance TYPE` - SageMaker processing instance type (default: ml.c5.xlarge)
- `--workshop-name NAME` - Name for workshop resources (default: model-optimization-workshop)

## Getting Started with the Workshop

After deploying the CloudFormation stack, follow these steps to set up the workshop environment:

1. **Access your SageMaker notebook instance**:
   - Open the SageMaker console
   - Find your notebook instance named "model-optimization-workshop-notebook"
   - Click "Open JupyterLab"

2. **Clone the workshop repository**:
   - Open a terminal in JupyterLab (File > New > Terminal)
   - Run the following commands:
     ```bash
     cd SageMaker
     git clone https://github.com/SaRedfiche/model-optimization-workshop.git
     cd model-optimization-workshop
     ```

3. **Install required packages**:
   - In the terminal, run:
     ```bash
     pip install -r requirements.txt
     ```

4. **Start the workshop**:
   - Navigate to the repository folder in the JupyterLab file browser
   - Open the first notebook: `01_introduction_and_setup.ipynb`
   - Follow the instructions in the notebook to configure your workshop settings
   - You'll need to enter the following values from the CloudFormation stack outputs:
     - S3 Bucket Name
     - AWS Region
     - SageMaker Role ARN
     - Optimization Instance Type

## Instance Type Recommendations

### Notebook Instance Types

- **ml.t3.medium** - Most cost-effective option for running the notebooks (2 vCPU, 4 GiB memory)
- **ml.t3.large** - Better performance for larger notebooks (2 vCPU, 8 GiB memory)
- **ml.t3.xlarge** - For more demanding notebook operations (4 vCPU, 16 GiB memory)

### Optimization Instance Types

- **ml.c5.xlarge** - Good for CPU-based quantization (4 vCPU, 8 GiB memory)
- **ml.c5.2xlarge** - Better for more demanding CPU tasks (8 vCPU, 16 GiB memory)
- **ml.g4dn.xlarge** - For GPU-accelerated tasks (4 vCPU, 16 GiB memory, 1 GPU)
- **ml.g4dn.2xlarge** - For more demanding GPU tasks (8 vCPU, 32 GiB memory, 1 GPU)

## Cleaning Up Resources

When you're done with the workshop, you can clean up all resources using the cleanup script:

1. Make the cleanup script executable:
   ```bash
   chmod +x cleanup-workshop.sh
   ```

2. Run the cleanup script:
   ```bash
   ./cleanup-workshop.sh
   ```

3. To empty the S3 bucket before deletion:
   ```bash
   ./cleanup-workshop.sh --empty-bucket
   ```

## Troubleshooting

- **Stack Creation Fails**: Check the CloudFormation events in the AWS Console for specific error messages
- **Notebook Instance Not Ready**: SageMaker notebook instances can take several minutes to provision
- **Permission Issues**: Ensure your AWS CLI user has sufficient permissions to create IAM roles and SageMaker resources
- **JupyterLab Not Loading**: Try accessing through the SageMaker console directly by clicking "Open JupyterLab"

# Model Optimization Workshop

This workshop provides a hands-on experience for optimizing machine learning models to improve performance and reduce costs. You'll learn various optimization techniques including quantization, pruning, knowledge distillation, and fine-tuning.

## Workshop Structure

1. **Introduction and Setup** - Environment setup and model download
2. **Quantization** - Applying quantization techniques
3. **Pruning** - Implementing pruning techniques
4. **Knowledge Distillation** - Creating smaller student models
5. **Fine-tuning** - Adapting models to specific tasks
6. **Cost Analysis** - Analyzing cost implications and ROI
7. **Resource Cleanup** - Cleaning up AWS resources

## Prerequisites

- AWS account with access to SageMaker
- Basic understanding of machine learning and deep learning concepts
- Familiarity with Python and PyTorch

## Setup Instructions

### Automated Setup with CloudFormation

The easiest way to set up the workshop environment is using the provided CloudFormation template:

1. Navigate to the `cloudformation` directory
2. Follow the instructions in the `README.md` file to deploy the CloudFormation stack
3. After the stack is deployed, follow the "Getting Started" instructions to clone the repository and set up the workshop environment

### Manual SageMaker Setup

If you prefer to set up the environment manually:

1. Launch a SageMaker notebook instance with the following specifications:
   - Instance type: ml.t3.xlarge (minimum) or ml.g4dn.xlarge (recommended for GPU acceleration)
   - Use the standard ML AMI provided by SageMaker
   - Attach an IAM role with the necessary permissions (AmazonSageMakerFullAccess and S3 access)

2. Clone this repository to your SageMaker notebook instance:
   ```bash
   git clone https://github.com/SaRedfiche/model-optimization-workshop.git
   cd model-optimization-workshop
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the first notebook `01_introduction_and_setup.ipynb` and follow the instructions.

### Local Setup (Alternative)

If you prefer to run the workshop locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/SaRedfiche/model-optimization-workshop.git
   cd model-optimization-workshop
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure AWS credentials:
   ```bash
   aws configure
   ```

5. Open the first notebook `01_introduction_and_setup.ipynb` and follow the instructions.

## Recommended Instance Types

- **Notebook Instance**:
  - **ml.t3.medium**: Good for initial notebooks and smaller models (4 vCPUs, 16 GB memory)
  - **ml.t3.large**: Better for more demanding notebooks (2 vCPU, 8 GiB memory)
  - **ml.g4dn.xlarge**: For GPU-accelerated tasks (4 vCPU, 16 GiB memory, 1 GPU)

- **Optimization Instance** (for distributed processing):
  - **ml.c5.xlarge**: Good for CPU-based quantization (4 vCPU, 8 GiB memory)
  - **ml.g4dn.xlarge**: For GPU-accelerated tasks (4 vCPU, 16 GiB memory, 1 GPU)

## Documentation

- `cloudformation/README.md` - Instructions for deploying with CloudFormation
- `model_optimization_techniques.md` - Overview of model optimization techniques
- `quantization_deep_dive.md` - Detailed explanation of quantization methods
- `pruning_techniques.md` - Detailed explanation of pruning techniques
- `cost_estimation_for_model_optimization.md` - Guide to estimating cost savings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

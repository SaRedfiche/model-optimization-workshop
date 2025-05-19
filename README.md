# Model Optimization Workshop

This workshop provides a hands-on experience for optimizing machine learning models to improve performance and reduce costs. You'll learn various optimization techniques including quantization, pruning, and knowledge distillation.

## Workshop Structure

1. **Introduction and Setup** - Environment setup and model download
2. **Baseline Evaluation** - Measuring baseline performance metrics
3. **Quantization** - Applying quantization techniques
4. **Pruning** - Implementing pruning techniques
5. **Knowledge Distillation** - Creating smaller student models
6. **Model Hosting** - Deploying models to AWS SageMaker
7. **Inference Performance** - Comparing inference performance across models
8. **Cost Analysis** - Analyzing cost implications and ROI
9. **Resource Cleanup** - Cleaning up AWS resources

## Prerequisites

- AWS account with access to SageMaker
- Basic understanding of machine learning and deep learning concepts
- Familiarity with Python and PyTorch

## Setup Instructions

### SageMaker Setup

1. Launch a SageMaker notebook instance with the following specifications:
   - Instance type: ml.t3.xlarge (minimum) or ml.g4dn.xlarge (recommended for GPU acceleration)
   - Use the standard ML AMI provided by SageMaker
   - Attach an IAM role with the necessary permissions (see `sagemaker_permissions.md`)

   **Recommended Instance Types:**
   - **ml.t3.xlarge**: Good for initial notebooks and smaller models (4 vCPUs, 16 GB memory)
   - **ml.g4dn.xlarge**: Better for optimization tasks with GPU acceleration (4 vCPUs, 16 GB memory, 1 NVIDIA T4 GPU)
   - **ml.g4dn.2xlarge**: For working with multiple large models (8 vCPUs, 32 GB memory, 1 NVIDIA T4 GPU)

2. Clone this repository to your SageMaker notebook instance:
   ```bash
   git clone https://github.com/SaRedfiche/model-optimization-workshop.git
   cd model-optimization-workshop
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the first notebook `01_introduction.ipynb` and follow the instructions.

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

5. Open the first notebook `01_introduction.ipynb` and follow the instructions.

## Documentation

- `sagemaker_permissions.md` - Details on the required IAM permissions
- `model_optimization_techniques.md` - Overview of model optimization techniques
- `quantization_deep_dive.md` - Detailed explanation of quantization methods
- `pruning_techniques.md` - Detailed explanation of pruning techniques
- `cost_estimation_for_model_optimization.md` - Guide to estimating cost savings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

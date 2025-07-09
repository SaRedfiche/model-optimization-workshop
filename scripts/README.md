# Training Scripts

This directory contains scripts used for fine-tuning models on SageMaker.

## `train_sentiment.py`

This script handles the fine-tuning of transformer models for sentiment analysis tasks. It's designed to work with the SageMaker distributed training environment and includes robust error handling and recovery mechanisms.

### Key Features

- **Distributed Training Support**: Automatically configures PyTorch Distributed Data Parallel (DDP) when running on multiple instances
- **GPU Memory Optimization**: Dynamically adjusts batch sizes based on available GPU memory
- **Error Recovery**: Automatically recovers from common training errors like CUDA out-of-memory issues
- **Comprehensive Logging**: Detailed logging of training progress, errors, and system information
- **Dataset Validation**: Validates dataset structure before training to catch issues early

### Usage

The script is designed to be used with SageMaker's HuggingFace estimator. It accepts the following hyperparameters:

- `epochs`: Number of training epochs
- `train-batch-size`: Batch size for training
- `eval-batch-size`: Batch size for evaluation
- `learning-rate`: Learning rate for optimization
- `warmup-steps`: Number of warmup steps for learning rate scheduler
- `model-id`: Hugging Face model ID to use for fine-tuning
- `fp16`: Whether to use mixed precision training (boolean)
- `gradient-accumulation-steps`: Number of steps to accumulate gradients before updating weights
- `auto-recover`: Whether to attempt recovery from training errors (boolean)

### Environment Variables

The script uses the following SageMaker environment variables:

- `SM_MODEL_DIR`: Directory to save the model
- `SM_CHANNEL_TRAIN`: Directory containing the training dataset
- `SM_CHANNEL_TEST`: Directory containing the test dataset
- `SM_OUTPUT_DATA_DIR`: Directory for output data
- `SM_HOSTS`: List of hosts in the training cluster
- `SM_CURRENT_HOST`: Current host name
- `LOCAL_RANK`: Local rank for distributed training

### Error Handling

The script includes robust error handling for common issues:

1. **CUDA Out of Memory**: Automatically reduces batch size and increases gradient accumulation steps
2. **Dataset Loading Errors**: Validates dataset structure and provides clear error messages
3. **Distributed Training Setup Issues**: Falls back to single-node training if distributed setup fails

### Output

The script saves the following outputs:

1. **Fine-tuned Model**: Saved to `SM_MODEL_DIR`
2. **Tokenizer**: Saved alongside the model
3. **Evaluation Results**: Saved as `eval_results.txt`
4. **Training Configuration**: Saved as `fine_tuning_config.json`
5. **Training Logs**: Available in CloudWatch and TensorBoard

### Customization

To customize the script for different tasks:

1. Modify the `compute_metrics` function for different evaluation metrics
2. Adjust the model loading code for different model architectures
3. Update the dataset loading code for different dataset structures

### Troubleshooting

If you encounter issues:

1. Check the CloudWatch logs for detailed error messages
2. Verify that your datasets have the required columns (`input_ids`, `attention_mask`, `label`)
3. Try reducing the batch size if you encounter memory issues
4. Ensure your model is compatible with the task (e.g., has the correct number of output classes)

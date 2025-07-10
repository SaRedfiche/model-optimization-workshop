#!/usr/bin/env python
# Fine-tuning solution for sentiment analysis using SageMaker and Hugging Face

import os
import argparse
import logging
import sys
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model for sentiment analysis")
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="distilbert-base-uncased",
        help="Pre-trained model ID to use for fine-tuning",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="glue",
        help="Dataset name to use for fine-tuning",
    )
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sst2",
        help="Dataset configuration to use for fine-tuning",
    )
    
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=True,
        help="S3 bucket to store datasets and model artifacts",
    )
    
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="sentiment-analysis",
        help="S3 prefix for datasets and model artifacts",
    )
    
    parser.add_argument(
        "--instance_type",
        type=str,
        default="ml.g4dn.xlarge",
        help="SageMaker instance type for training",
    )
    
    parser.add_argument(
        "--instance_count",
        type=int,
        default=1,
        help="Number of instances to use for training",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    
    return parser.parse_args()

def prepare_dataset(dataset_name, dataset_config, model_id, max_length=128):
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    logger.info(f"Loading tokenizer for model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    logger.info("Tokenizing dataset")
    tokenized_datasets = {}
    
    # Process training split
    if "train" in dataset:
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            # Keep the labels
            tokenized["labels"] = examples["label"]
            return tokenized
        
        tokenized_datasets["train"] = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence"],  # Only remove sentence column, keep label
        )
    
    # Process validation split
    if "validation" in dataset:
        def tokenize_function_val(examples):
            tokenized = tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            # Keep the labels
            tokenized["labels"] = examples["label"]
            return tokenized
        
        tokenized_datasets["validation"] = dataset["validation"].map(
            tokenize_function_val,
            batched=True,
            remove_columns=["sentence"],  # Only remove sentence column, keep label
        )
    
    return tokenized_datasets

def upload_to_s3(dataset_dict, s3_bucket, s3_prefix):
    logger.info("Saving datasets locally")
    os.makedirs("data", exist_ok=True)
    
    s3_paths = {}
    for split_name, dataset in dataset_dict.items():
        local_path = f"data/{split_name}"
        dataset.save_to_disk(local_path)
        
        logger.info(f"Uploading {split_name} dataset to S3")
        s3_path = f"s3://{s3_bucket}/{s3_prefix}/data/{split_name}"
        
        # Use AWS CLI for uploading
        os.system(f"aws s3 cp {local_path} {s3_path} --recursive")
        
        logger.info(f"Dataset uploaded to {s3_path}")
        s3_paths[split_name] = s3_path
    
    return s3_paths

def create_training_script():
    """Create the training script for SageMaker"""
    os.makedirs("scripts", exist_ok=True)
    
    script_content = """#!/usr/bin/env python
# Training script for sentiment analysis fine-tuning

import argparse
import logging
import os
import sys
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def compute_metrics(pred):
    \"\"\"
    Compute metrics for evaluation.
    \"\"\"
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_args():
    \"\"\"
    Parse arguments for training.
    \"\"\"
    parser = argparse.ArgumentParser()
    
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    
    args, _ = parser.parse_known_args()
    return args

def main():
    \"\"\"
    Main training function.
    \"\"\"
    args = parse_args()
    
    # Load datasets
    logger.info(f"Loading datasets from {args.training_dir} and {args.validation_dir}")
    train_dataset = load_from_disk(args.training_dir)
    validation_dataset = load_from_disk(args.validation_dir)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(validation_dataset)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=len(train_dataset.unique("labels"))
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate(eval_dataset=validation_dataset)
    logger.info(f"Evaluation results: {eval_result}")
    
    # Write eval results to file
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        for key, value in eval_result.items():
            writer.write(f"{key} = {value}\\n")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save special tokens file
    special_tokens_map_file = os.path.join(args.model_dir, "special_tokens_map.json")
    if not os.path.exists(special_tokens_map_file):
        with open(special_tokens_map_file, "w") as f:
            f.write("{}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
"""
    
    with open("scripts/train.py", "w") as f:
        f.write(script_content)
    
    logger.info("Created training script at scripts/train.py")

def main():
    args = parse_args()
    
    # Set up S3 paths
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix
    output_path = f"s3://{s3_bucket}/{s3_prefix}/output"
    
    logger.info(f"Using model: {args.model_id}")
    logger.info(f"Using dataset: {args.dataset_name}/{args.dataset_config}")
    logger.info(f"S3 bucket: {s3_bucket}")
    logger.info(f"S3 prefix: {s3_prefix}")
    
    # Prepare and upload dataset
    tokenized_datasets = prepare_dataset(args.dataset_name, args.dataset_config, args.model_id)
    s3_paths = upload_to_s3(tokenized_datasets, s3_bucket, s3_prefix)
    
    train_data_path = s3_paths.get("train")
    validation_data_path = s3_paths.get("validation")
    
    if not train_data_path or not validation_data_path:
        logger.error("Failed to prepare and upload datasets")
        sys.exit(1)
    
    # Create training script
    create_training_script()
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': args.epochs,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_name': args.model_id,
    }
    
    # Define metric definitions for tracking
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
        {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'},
        {'Name': 'eval:accuracy', 'Regex': 'eval_accuracy: ([0-9\\.]+)'},
        {'Name': 'eval:f1', 'Regex': 'eval_f1: ([0-9\\.]+)'},
    ]
    
    # Create Hugging Face estimator with supported versions
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role=role,
        transformers_version='4.49.0',
        pytorch_version='2.6.0',
        py_version='py311',
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        output_path=output_path
    )
    
    # Define data channels
    data_channels = {
        'train': train_data_path,
        'validation': validation_data_path
    }
    
    # Start training job
    logger.info("Starting training job...")
    huggingface_estimator.fit(data_channels, wait=True)
    
    logger.info(f"Training job completed. Model artifacts saved to: {huggingface_estimator.model_data}")
    
    return huggingface_estimator.model_data

if __name__ == "__main__":
    main()

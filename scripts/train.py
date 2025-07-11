#!/usr/bin/env python
# Training script for sentiment analysis fine-tuning

import argparse
import logging
import os
import sys
import numpy as np
import torch
from datasets import load_from_disk, Dataset
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
    """
    Compute metrics for evaluation.
    """
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
    """
    Parse arguments for training.
    """
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
    """
    Main training function.
    """
    args = parse_args()
    
    # Load datasets
    logger.info(f"Loading datasets from {args.training_dir} and {args.validation_dir}")
    try:
        train_dataset = load_from_disk(args.training_dir)
        validation_dataset = load_from_disk(args.validation_dir)
    except Exception as e:
        logger.error(f"Error loading datasets with load_from_disk: {e}")
        logger.info("Attempting to load datasets as individual files...")
        
        # Try loading as individual files if load_from_disk fails
        import json
        
        # Load training data
        train_files = [f for f in os.listdir(args.training_dir) if f.endswith('.json')]
        if train_files:
            with open(os.path.join(args.training_dir, train_files[0]), 'r') as f:
                train_data = json.load(f)
            train_dataset = Dataset.from_list(train_data)
        else:
            raise ValueError(f"No JSON files found in {args.training_dir}")
        
        # Load validation data
        val_files = [f for f in os.listdir(args.validation_dir) if f.endswith('.json')]
        if val_files:
            with open(os.path.join(args.validation_dir, val_files[0]), 'r') as f:
                val_data = json.load(f)
            validation_dataset = Dataset.from_list(val_data)
        else:
            raise ValueError(f"No JSON files found in {args.validation_dir}")
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(validation_dataset)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get number of unique labels
    try:
        num_labels = len(train_dataset.unique("labels"))
    except:
        # Fallback: assume binary classification
        num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
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
        save_strategy="epoch",
        logging_steps=10,
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
            writer.write(f"{key} = {value}\n")
    
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

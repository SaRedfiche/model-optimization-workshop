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
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--model-id", type=str, default="distilbert-base-uncased")
    parser.add_argument("--fp16", type=bool, default=True)
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info(f"Loading datasets from {args.training_dir} and {args.test_dir}")
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, 
        num_labels=2  # Binary classification
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Set up data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Set up distributed training
    is_distributed = len(os.environ.get("SM_HOSTS", [])) > 1
    if is_distributed:
        logger.info("Distributed training enabled")
        world_size = int(os.environ.get("SM_NUM_GPUS", 1)) * len(os.environ.get("SM_HOSTS", []))
        logger.info(f"World size: {world_size}")
    else:
        logger.info("Distributed training not enabled")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=100,
        fp16=args.fp16,  # Enable mixed precision training
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,  # Only keep the 2 best checkpoints
        report_to="tensorboard",
        # Distributed training settings
        ddp_find_unused_parameters=False if is_distributed else None,
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # Save the model
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save evaluation results
    with open(os.path.join(args.model_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_result.items():
            f.write(f"{key} = {value}\n")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

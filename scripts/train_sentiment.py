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
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--training-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "./data/test"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--model-id", type=str, default="distilbert-base-uncased")
    parser.add_argument("--fp16", type=bool, default=True)
    
    return parser.parse_args()

def setup_distributed_training():
    """Set up distributed training environment."""
    # Check if running in SageMaker
    if "SM_HOSTS" in os.environ:
        # Get SageMaker environment variables
        hosts = os.environ.get("SM_HOSTS", "").split(",")
        current_host = os.environ.get("SM_CURRENT_HOST", "")
        rank = hosts.index(current_host)
        world_size = len(hosts)
        
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = hosts[0]
        os.environ["MASTER_PORT"] = "29500"
        
        # Log distributed training setup
        logger.info(f"Distributed training setup: rank={rank}, world_size={world_size}")
        logger.info(f"Master: {hosts[0]}:{os.environ['MASTER_PORT']}")
        
        return True, world_size
    else:
        logger.info("Not running in SageMaker, skipping distributed setup")
        return False, 1

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up distributed training
    is_distributed, world_size = setup_distributed_training()
    
    # Load datasets
    logger.info(f"Loading datasets from {args.training_dir} and {args.test_dir}")
    try:
        train_dataset = load_from_disk(args.training_dir)
        test_dataset = load_from_disk(args.test_dir)
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_id}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, 
            num_labels=2  # Binary classification
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
    
    # Set up data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
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
        local_rank=int(os.environ.get("LOCAL_RANK", -1)) if is_distributed else -1,
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
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    # Evaluate the model
    logger.info("Evaluating model...")
    try:
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        eval_result = {"error": str(e)}
    
    # Save the model
    logger.info(f"Saving model to {args.model_dir}")
    try:
        trainer.save_model(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
    
    # Save evaluation results
    try:
        with open(os.path.join(args.model_dir, "eval_results.txt"), "w") as f:
            for key, value in eval_result.items():
                f.write(f"{key} = {value}\n")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

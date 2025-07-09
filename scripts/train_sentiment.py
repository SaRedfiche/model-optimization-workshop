#!/usr/bin/env python
# coding=utf-8

"""
Fine-tuning script for Hugging Face Transformers models on sentiment analysis tasks.
Adapted for SageMaker distributed training with robust error handling.
"""

# First, ensure we have compatible versions of numpy and pandas
import subprocess
import sys

def install_requirements():
    print("Installing required packages with compatible versions...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.22.4"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas>=1.5.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.0.0"])
    print("Required packages installed successfully.")

# Install requirements before importing other packages
install_requirements()

import argparse
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Will error if the minimal version of Transformers is not installed
check_min_version("4.20.0")

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_id: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    auto_recover: bool = field(
        default=False,
        metadata={"help": "Whether to automatically recover from errors during training."},
    )


def compute_metrics(p: EvalPrediction):
    """
    Compute metrics for evaluation.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    # Calculate precision, recall, f1, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def setup_logging(is_main_process):
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process else logging.WARN)


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(training_args.local_rank <= 0)
    
    # Log on each process the small summary:
    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Load datasets
    try:
        logger.info("Loading datasets from disk")
        train_dataset = load_from_disk(os.path.join(os.environ["SM_CHANNEL_TRAIN"]))
        eval_dataset = load_from_disk(os.path.join(os.environ["SM_CHANNEL_TEST"]))
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # Log a few examples
        for i in range(min(3, len(train_dataset))):
            logger.info(f"Train example {i}: {train_dataset[i]}")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        if model_args.auto_recover:
            logger.info("Attempting to recover by loading datasets differently...")
            try:
                # Try alternative loading method
                train_dataset = datasets.load_from_disk(os.environ["SM_CHANNEL_TRAIN"])
                eval_dataset = datasets.load_from_disk(os.environ["SM_CHANNEL_TEST"])
            except Exception as e2:
                logger.error(f"Recovery failed: {e2}")
                raise e2
        else:
            raise e

    # Load pretrained model and tokenizer
    logger.info(f"Loading model: {model_args.model_id}")
    
    try:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_id,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_id,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_id,
            from_tf=bool(".ckpt" in model_args.model_id),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if model_args.auto_recover:
            logger.info("Attempting to recover by loading model with different options...")
            try:
                # Try with minimal options
                model = AutoModelForSequenceClassification.from_pretrained(model_args.model_id)
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
                config = model.config
            except Exception as e2:
                logger.error(f"Recovery failed: {e2}")
                raise e2
        else:
            raise e

    # Resize token embeddings if needed
    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(
            f"The tokenizer has a vocab size of {tokenizer.vocab_size} while the model has {model.config.vocab_size}. "
            "Resizing model embeddings to match tokenizer."
        )
        model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    try:
        logger.info("*** Starting training ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if model_args.auto_recover:
            logger.info("Attempting to recover and continue training...")
            try:
                # Try to continue from last checkpoint
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                if last_checkpoint:
                    logger.info(f"Resuming from checkpoint: {last_checkpoint}")
                    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
                    metrics = train_result.metrics
                    trainer.save_model()
                    trainer.log_metrics("train", metrics)
                    trainer.save_metrics("train", metrics)
                    trainer.save_state()
                else:
                    logger.error("No checkpoint found, cannot recover")
                    raise e
            except Exception as e2:
                logger.error(f"Recovery failed: {e2}")
                raise e2
        else:
            raise e

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save to model directory for SageMaker
    if training_args.local_rank <= 0:  # Only save on main process
        final_model_path = os.path.join(os.environ["SM_MODEL_DIR"], "final")
        logger.info(f"Saving model to {final_model_path}")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)


if __name__ == "__main__":
    main()

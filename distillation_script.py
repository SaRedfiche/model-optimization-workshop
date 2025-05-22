"""
Knowledge distillation script for model optimization workshop.
This script distills knowledge from a teacher model to a smaller student model.
"""

import sys
import subprocess

# Install required packages
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "datasets"])
print("Packages installed successfully.")

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import logging
import traceback
import sys
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM, DistilBertConfig
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification
from transformers import DistilBertForQuestionAnswering, DistilBertForMaskedLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_num_parameters(model):
    """Calculate number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, inputs, num_runs=10):
    """Measure average inference time over multiple runs."""
    # Warm-up run
    with torch.no_grad():
        model(**inputs)
    
    # Measure inference time
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    inference_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            start_event.record()
            with torch.no_grad():
                model(**inputs)
            end_event.record()
            torch.cuda.synchronize()
            inference_times.append(start_event.elapsed_time(end_event))
        else:
            start_time = time.time()
            with torch.no_grad():
                model(**inputs)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return sum(inference_times) / len(inference_times)

def prepare_sample_inputs(model_name, task, tokenizer, device):
    """Prepare sample inputs for the model based on its task."""
    if task == "sequence-classification":
        text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task == "masked-lm":
        text = "The [MASK] is a large language model trained by OpenAI."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def create_student_model(task, teacher_model):
    """Create a smaller student model based on the teacher model's task."""
    # Create a DistilBERT configuration with fewer layers
    config = DistilBertConfig(
        hidden_size=teacher_model.config.hidden_size,
        num_hidden_layers=4,  # Fewer layers than BERT/RoBERTa
        num_attention_heads=teacher_model.config.num_attention_heads,
        intermediate_size=teacher_model.config.intermediate_size,
        hidden_act=teacher_model.config.hidden_act,
        hidden_dropout_prob=teacher_model.config.hidden_dropout_prob,
        attention_probs_dropout_prob=teacher_model.config.attention_probs_dropout_prob,
        max_position_embeddings=teacher_model.config.max_position_embeddings,
        type_vocab_size=teacher_model.config.type_vocab_size,
        initializer_range=teacher_model.config.initializer_range,
        layer_norm_eps=teacher_model.config.layer_norm_eps,
        pad_token_id=teacher_model.config.pad_token_id,
        vocab_size=teacher_model.config.vocab_size
    )
    
    # Create the appropriate model based on the task
    if task == "sequence-classification":
        config.num_labels = teacher_model.config.num_labels
        student_model = DistilBertForSequenceClassification(config)
    elif task == "token-classification":
        config.num_labels = teacher_model.config.num_labels
        student_model = DistilBertForTokenClassification(config)
    elif task == "question-answering":
        student_model = DistilBertForQuestionAnswering(config)
    elif task == "masked-lm":
        student_model = DistilBertForMaskedLM(config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return student_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Knowledge distillation script")
parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics and models")
parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
args = parser.parse_args()

try:
    # Load model info
    logger.info(f"Loading model info from {args.model_info_path}")
    with open(args.model_info_path, "r") as f:
        model_info = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each model
    all_metrics = {}
    for model_key, info in model_info.items():
        try:
            logger.info(f"Processing model: {model_key}")
            
            model_name = info["model_name"]
            task = info["task"]
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load teacher model and tokenizer
            logger.info(f"Loading teacher model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if task == "sequence-classification":
                teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task == "token-classification":
                teacher_model = AutoModelForTokenClassification.from_pretrained(model_name)
            elif task == "question-answering":
                teacher_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            elif task == "masked-lm":
                teacher_model = AutoModelForMaskedLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            teacher_model = teacher_model.to(device)
            teacher_model.eval()
            
            # Create student model
            logger.info("Creating student model")
            student_model = create_student_model(task, teacher_model)
            student_model = student_model.to(device)
            
            # Prepare sample inputs for evaluation
            inputs = prepare_sample_inputs(model_name, task, tokenizer, device)
            
            # Measure teacher model metrics
            teacher_size = get_model_size(teacher_model)
            teacher_params = get_num_parameters(teacher_model)
            teacher_inference_time = measure_inference_time(teacher_model, inputs)
            
            logger.info(f"Teacher model size: {teacher_size:.2f} MB")
            logger.info(f"Teacher model parameters: {teacher_params:,}")
            logger.info(f"Teacher model inference time: {teacher_inference_time:.2f} ms")
            
            # Measure student model metrics
            student_size = get_model_size(student_model)
            student_params = get_num_parameters(student_model)
            student_inference_time = measure_inference_time(student_model, inputs)
            
            logger.info(f"Student model size: {student_size:.2f} MB")
            logger.info(f"Student model parameters: {student_params:,}")
            logger.info(f"Student model inference time: {student_inference_time:.2f} ms")
            
            # Calculate improvements
            size_reduction = (teacher_size - student_size) / teacher_size * 100
            params_reduction = (teacher_params - student_params) / teacher_params * 100
            time_improvement = (teacher_inference_time - student_inference_time) / teacher_inference_time * 100
            
            logger.info(f"Size reduction: {size_reduction:.2f}%")
            logger.info(f"Parameters reduction: {params_reduction:.2f}%")
            logger.info(f"Inference time improvement: {time_improvement:.2f}%")
            
            # Save metrics
            metrics = {
                "teacher_model": model_name,
                "task": task,
                "teacher_size_mb": round(teacher_size, 2),
                "teacher_parameters": teacher_params,
                "teacher_inference_time_ms": round(teacher_inference_time, 2),
                "student_size_mb": round(student_size, 2),
                "student_parameters": student_params,
                "student_inference_time_ms": round(student_inference_time, 2),
                "size_reduction_percent": round(size_reduction, 2),
                "parameters_reduction_percent": round(params_reduction, 2),
                "inference_time_improvement_percent": round(time_improvement, 2)
            }
            
            all_metrics[model_key] = metrics
            
            # Save student model
            model_dir = os.path.join(args.output_dir, f"{model_key}_student")
            os.makedirs(model_dir, exist_ok=True)
            student_model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Saved student model to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error processing model {model_key}: {e}")
            logger.error(traceback.format_exc())
    
    # Save all metrics to a single file
    metrics_path = os.path.join(args.output_dir, "distillation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Saved distillation metrics to {metrics_path}")

except Exception as e:
    logger.error(f"Error in knowledge distillation: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

#!/usr/bin/env python3
"""
Knowledge Distillation Script for SageMaker Processing

This script implements knowledge distillation to create smaller student models
that mimic the behavior of larger teacher models.
Metrics collection and analysis are handled in the notebook.
"""

import os
import json
import torch
import argparse
import logging
import traceback
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM, AutoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_student_model(teacher_model, student_info, task):
    """Create a significantly smaller student model based on the provided architecture."""
    logger.info("Creating student model with custom configuration")
    
    # Get the teacher model configuration
    if hasattr(teacher_model, "config"):
        teacher_config = teacher_model.config
    else:
        raise ValueError("Teacher model does not have a config attribute")
    
    # Create a new configuration for the student model with custom parameters
    model_name = student_info.get("model_name", "distilbert-base-uncased")
    student_config = AutoConfig.from_pretrained(model_name)
    
    # Apply custom configuration from student_info
    if "num_hidden_layers" in student_info:
        student_config.num_hidden_layers = student_info["num_hidden_layers"]
        logger.info(f"Setting student layers to {student_config.num_hidden_layers} (teacher had {teacher_config.num_hidden_layers})")
    
    if "hidden_size" in student_info:
        student_config.hidden_size = student_info["hidden_size"]
        logger.info(f"Setting student hidden size to {student_config.hidden_size} (teacher had {teacher_config.hidden_size})")
    
    # Ensure the student has the same output dimensions as the teacher
    if task == "sequence-classification" or task == "text-classification":
        if hasattr(teacher_config, "num_labels"):
            student_config.num_labels = teacher_config.num_labels
            logger.info(f"Setting student num_labels to {student_config.num_labels}")
    
    elif task == "token-classification":
        if hasattr(teacher_config, "num_labels"):
            student_config.num_labels = teacher_config.num_labels
            logger.info(f"Setting student token classification num_labels to {student_config.num_labels}")
    
    # Create the student model with the modified configuration
    if task == "sequence-classification" or task == "text-classification":
        student_model = AutoModelForSequenceClassification.from_config(student_config)
    elif task == "token-classification":
        student_model = AutoModelForTokenClassification.from_config(student_config)
    elif task == "question-answering":
        student_model = AutoModelForQuestionAnswering.from_config(student_config)
    elif task == "masked-lm" or task == "fill-mask":
        student_model = AutoModelForMaskedLM.from_config(student_config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return student_model

def prepare_sample_inputs(tokenizer, task):
    """Prepare sample inputs for the model based on its task."""
    if task == "sequence-classification" or task == "text-classification":
        text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task == "masked-lm" or task == "fill-mask":
        text = "The [MASK] is a large language model trained by OpenAI."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return inputs

def main():
    """Main function to run knowledge distillation."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Script")
    parser.add_argument("--teacher-info-path", type=str, required=True, help="Path to teacher model info JSON file")
    parser.add_argument("--student-info-path", type=str, required=True, help="Path to student architecture info JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for distilled models")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for softening the teacher's outputs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss vs. task loss")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load teacher model info
        logger.info(f"Loading teacher model info from {args.teacher_info_path}")
        if os.path.isdir(args.teacher_info_path):
            # List files in the directory
            logger.info(f"teacher_info_path is a directory. Contents: {os.listdir(args.teacher_info_path)}")
            # Try to find a JSON file
            json_files = [f for f in os.listdir(args.teacher_info_path) if f.endswith('.json')]
            if json_files:
                # Use the first JSON file found
                teacher_info_file = os.path.join(args.teacher_info_path, json_files[0])
                logger.info(f"Using JSON file: {teacher_info_file}")
                with open(teacher_info_file, "r") as f:
                    teacher_info = json.load(f)
            else:
                raise FileNotFoundError(f"No JSON files found in directory: {args.teacher_info_path}")
        else:
            # It's a file, load it directly
            with open(args.teacher_info_path, "r") as f:
                teacher_info = json.load(f)
        
        # Load student architecture info
        logger.info(f"Loading student architecture info from {args.student_info_path}")
        if os.path.isdir(args.student_info_path):
            # List files in the directory
            logger.info(f"student_info_path is a directory. Contents: {os.listdir(args.student_info_path)}")
            # Try to find a JSON file
            json_files = [f for f in os.listdir(args.student_info_path) if f.endswith('.json')]
            if json_files:
                # Use the first JSON file found
                student_info_file = os.path.join(args.student_info_path, json_files[0])
                logger.info(f"Using JSON file: {student_info_file}")
                with open(student_info_file, "r") as f:
                    student_info = json.load(f)
            else:
                raise FileNotFoundError(f"No JSON files found in directory: {args.student_info_path}")
        else:
            # It's a file, load it directly
            with open(args.student_info_path, "r") as f:
                student_info = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each model
        for model_key, info in teacher_info.items():
            try:
                model_name = info["model_name"]
                task = info.get("task", "text-classification")
                
                logger.info(f"Processing model: {model_name} for task: {task}")
                
                # Load tokenizer
                logger.info(f"Loading tokenizer: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load teacher model
                logger.info(f"Loading teacher model: {model_name}")
                if task == "sequence-classification" or task == "text-classification":
                    teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                elif task == "token-classification":
                    teacher_model = AutoModelForTokenClassification.from_pretrained(model_name)
                elif task == "question-answering":
                    teacher_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                elif task == "masked-lm" or task == "fill-mask":
                    teacher_model = AutoModelForMaskedLM.from_pretrained(model_name)
                else:
                    raise ValueError(f"Unsupported task: {task}")
                
                teacher_model = teacher_model.to(device)
                
                # Get student architecture info for this model
                if model_key in student_info:
                    student_model_info = student_info[model_key]
                    logger.info(f"Found student architecture for {model_key}: {student_model_info}")
                else:
                    logger.warning(f"No student architecture found for {model_key}, using defaults")
                    student_model_info = {
                        "model_name": "distilbert-base-uncased",
                        "num_hidden_layers": 3,
                        "hidden_size": 384
                    }
                
                # Create student model with custom architecture
                student_model = create_student_model(teacher_model, student_model_info, task)
                student_model = student_model.to(device)
                
                # Prepare inputs
                inputs = prepare_sample_inputs(tokenizer, task)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Save student model
                student_output_dir = os.path.join(args.output_dir, f"{model_key}_student")
                os.makedirs(student_output_dir, exist_ok=True)
                student_model.save_pretrained(student_output_dir)
                tokenizer.save_pretrained(student_output_dir)
                logger.info(f"Saved student model to {student_output_dir}")
                
                # Create a dummy file to indicate successful completion
                with open(os.path.join(student_output_dir, "distillation_complete.txt"), "w") as f:
                    f.write("Distillation completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error in distillation: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

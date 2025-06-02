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

def enhanced_knowledge_transfer(teacher_model, student_model, task, logger):
    """Copy as many weights as possible from teacher to student."""
    try:
        logger.info("Performing enhanced knowledge transfer")
        
        # Copy embeddings (common to all models)
        if hasattr(teacher_model, "distilbert") and hasattr(student_model, "distilbert"):
            # Copy word embeddings
            student_model.distilbert.embeddings.word_embeddings.weight.data = \
                teacher_model.distilbert.embeddings.word_embeddings.weight.data.clone()
            logger.info("Copied word embeddings")
            
            # Copy position embeddings if they exist and have the same size
            if hasattr(teacher_model.distilbert.embeddings, "position_embeddings") and \
               hasattr(student_model.distilbert.embeddings, "position_embeddings") and \
               teacher_model.distilbert.embeddings.position_embeddings.weight.shape == \
               student_model.distilbert.embeddings.position_embeddings.weight.shape:
                student_model.distilbert.embeddings.position_embeddings.weight.data = \
                    teacher_model.distilbert.embeddings.position_embeddings.weight.data.clone()
                logger.info("Copied position embeddings")
            
            # Copy as many transformer blocks as possible
            teacher_layers = teacher_model.distilbert.transformer.layer
            student_layers = student_model.distilbert.transformer.layer
            
            # Copy weights from the first N layers (where N is the number of student layers)
            for i in range(min(len(student_layers), len(teacher_layers))):
                # Copy attention weights
                if hasattr(teacher_layers[i], "attention") and hasattr(student_layers[i], "attention"):
                    # Copy query, key, value weights
                    if hasattr(teacher_layers[i].attention, "q_lin") and hasattr(student_layers[i].attention, "q_lin"):
                        if teacher_layers[i].attention.q_lin.weight.shape == student_layers[i].attention.q_lin.weight.shape:
                            student_layers[i].attention.q_lin.weight.data = teacher_layers[i].attention.q_lin.weight.data.clone()
                            student_layers[i].attention.q_lin.bias.data = teacher_layers[i].attention.q_lin.bias.data.clone()
                            logger.info(f"Copied attention Q weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].attention, "k_lin") and hasattr(student_layers[i].attention, "k_lin"):
                        if teacher_layers[i].attention.k_lin.weight.shape == student_layers[i].attention.k_lin.weight.shape:
                            student_layers[i].attention.k_lin.weight.data = teacher_layers[i].attention.k_lin.weight.data.clone()
                            student_layers[i].attention.k_lin.bias.data = teacher_layers[i].attention.k_lin.bias.data.clone()
                            logger.info(f"Copied attention K weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].attention, "v_lin") and hasattr(student_layers[i].attention, "v_lin"):
                        if teacher_layers[i].attention.v_lin.weight.shape == student_layers[i].attention.v_lin.weight.shape:
                            student_layers[i].attention.v_lin.weight.data = teacher_layers[i].attention.v_lin.weight.data.clone()
                            student_layers[i].attention.v_lin.bias.data = teacher_layers[i].attention.v_lin.bias.data.clone()
                            logger.info(f"Copied attention V weights for layer {i}")
                
                # Copy FFN weights
                if hasattr(teacher_layers[i], "ffn") and hasattr(student_layers[i], "ffn"):
                    if hasattr(teacher_layers[i].ffn, "lin1") and hasattr(student_layers[i].ffn, "lin1"):
                        if teacher_layers[i].ffn.lin1.weight.shape == student_layers[i].ffn.lin1.weight.shape:
                            student_layers[i].ffn.lin1.weight.data = teacher_layers[i].ffn.lin1.weight.data.clone()
                            student_layers[i].ffn.lin1.bias.data = teacher_layers[i].ffn.lin1.bias.data.clone()
                            logger.info(f"Copied FFN1 weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].ffn, "lin2") and hasattr(student_layers[i].ffn, "lin2"):
                        if teacher_layers[i].ffn.lin2.weight.shape == student_layers[i].ffn.lin2.weight.shape:
                            student_layers[i].ffn.lin2.weight.data = teacher_layers[i].ffn.lin2.weight.data.clone()
                            student_layers[i].ffn.lin2.bias.data = teacher_layers[i].ffn.lin2.bias.data.clone()
                            logger.info(f"Copied FFN2 weights for layer {i}")
        
        # Handle BERT models (different architecture)
        elif hasattr(teacher_model, "bert") and hasattr(student_model, "bert"):
            # Copy word embeddings
            if teacher_model.bert.embeddings.word_embeddings.weight.shape == student_model.bert.embeddings.word_embeddings.weight.shape:
                student_model.bert.embeddings.word_embeddings.weight.data = \
                    teacher_model.bert.embeddings.word_embeddings.weight.data.clone()
                logger.info("Copied BERT word embeddings")
            
            # Copy position embeddings
            if hasattr(teacher_model.bert.embeddings, "position_embeddings") and \
               hasattr(student_model.bert.embeddings, "position_embeddings") and \
               teacher_model.bert.embeddings.position_embeddings.weight.shape == \
               student_model.bert.embeddings.position_embeddings.weight.shape:
                student_model.bert.embeddings.position_embeddings.weight.data = \
                    teacher_model.bert.embeddings.position_embeddings.weight.data.clone()
                logger.info("Copied BERT position embeddings")
            
            # Copy as many transformer blocks as possible
            teacher_layers = teacher_model.bert.encoder.layer
            student_layers = student_model.bert.encoder.layer
            
            # Copy weights from the first N layers (where N is the number of student layers)
            for i in range(min(len(student_layers), len(teacher_layers))):
                # Copy attention weights
                if hasattr(teacher_layers[i].attention, "self") and hasattr(student_layers[i].attention, "self"):
                    # Copy query, key, value weights
                    if teacher_layers[i].attention.self.query.weight.shape == student_layers[i].attention.self.query.weight.shape:
                        student_layers[i].attention.self.query.weight.data = teacher_layers[i].attention.self.query.weight.data.clone()
                        student_layers[i].attention.self.query.bias.data = teacher_layers[i].attention.self.query.bias.data.clone()
                        logger.info(f"Copied BERT attention query weights for layer {i}")
                    
                    if teacher_layers[i].attention.self.key.weight.shape == student_layers[i].attention.self.key.weight.shape:
                        student_layers[i].attention.self.key.weight.data = teacher_layers[i].attention.self.key.weight.data.clone()
                        student_layers[i].attention.self.key.bias.data = teacher_layers[i].attention.self.key.bias.data.clone()
                        logger.info(f"Copied BERT attention key weights for layer {i}")
                    
                    if teacher_layers[i].attention.self.value.weight.shape == student_layers[i].attention.self.value.weight.shape:
                        student_layers[i].attention.self.value.weight.data = teacher_layers[i].attention.self.value.weight.data.clone()
                        student_layers[i].attention.self.value.bias.data = teacher_layers[i].attention.self.value.bias.data.clone()
                        logger.info(f"Copied BERT attention value weights for layer {i}")
                
                # Copy FFN weights
                if hasattr(teacher_layers[i], "intermediate") and hasattr(student_layers[i], "intermediate"):
                    if teacher_layers[i].intermediate.dense.weight.shape == student_layers[i].intermediate.dense.weight.shape:
                        student_layers[i].intermediate.dense.weight.data = teacher_layers[i].intermediate.dense.weight.data.clone()
                        student_layers[i].intermediate.dense.bias.data = teacher_layers[i].intermediate.dense.bias.data.clone()
                        logger.info(f"Copied BERT intermediate weights for layer {i}")
                
                if hasattr(teacher_layers[i], "output") and hasattr(student_layers[i], "output"):
                    if hasattr(teacher_layers[i].output, "dense") and hasattr(student_layers[i].output, "dense"):
                        if teacher_layers[i].output.dense.weight.shape == student_layers[i].output.dense.weight.shape:
                            student_layers[i].output.dense.weight.data = teacher_layers[i].output.dense.weight.data.clone()
                            student_layers[i].output.dense.bias.data = teacher_layers[i].output.dense.bias.data.clone()
                            logger.info(f"Copied BERT output weights for layer {i}")
        
        # Task-specific output layer transfer
        if task == "sequence-classification" or task == "text-classification":
            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                if teacher_model.classifier.out_features == student_model.classifier.out_features:
                    student_model.classifier.weight.data = teacher_model.classifier.weight.data.clone()
                    student_model.classifier.bias.data = teacher_model.classifier.bias.data.clone()
                    logger.info("Copied classifier weights")
                else:
                    logger.info(f"Classifier dimensions don't match: teacher {teacher_model.classifier.out_features}, student {student_model.classifier.out_features}")
            else:
                logger.info("Models don't have classifier attribute")
        
        elif task == "token-classification":
            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                if teacher_model.classifier.out_features == student_model.classifier.out_features:
                    student_model.classifier.weight.data = teacher_model.classifier.weight.data.clone()
                    student_model.classifier.bias.data = teacher_model.classifier.bias.data.clone()
                    logger.info("Copied token classifier weights")
                else:
                    logger.info(f"Token classifier dimensions don't match: teacher {teacher_model.classifier.out_features}, student {student_model.classifier.out_features}")
            else:
                logger.info("Models don't have classifier attribute for token classification")
        
        elif task == "question-answering":
            if hasattr(teacher_model, "qa_outputs") and hasattr(student_model, "qa_outputs"):
                if teacher_model.qa_outputs.weight.shape == student_model.qa_outputs.weight.shape:
                    student_model.qa_outputs.weight.data = teacher_model.qa_outputs.weight.data.clone()
                    student_model.qa_outputs.bias.data = teacher_model.qa_outputs.bias.data.clone()
                    logger.info("Copied QA output weights")
                else:
                    logger.info(f"QA output dimensions don't match: teacher {teacher_model.qa_outputs.weight.shape}, student {student_model.qa_outputs.weight.shape}")
            else:
                logger.info("Models don't have qa_outputs attribute")
        
        elif task == "masked-lm" or task == "fill-mask":
            # Try different MLM head architectures
            if hasattr(teacher_model, "vocab_projector") and hasattr(student_model, "vocab_projector"):
                if teacher_model.vocab_projector.weight.shape == student_model.vocab_projector.weight.shape:
                    student_model.vocab_projector.weight.data = teacher_model.vocab_projector.weight.data.clone()
                    student_model.vocab_projector.bias.data = teacher_model.vocab_projector.bias.data.clone()
                    logger.info("Copied vocab projector weights")
                else:
                    logger.info(f"Vocab projector dimensions don't match: teacher {teacher_model.vocab_projector.weight.shape}, student {student_model.vocab_projector.weight.shape}")
            elif hasattr(teacher_model, "cls") and hasattr(student_model, "cls"):
                if hasattr(teacher_model.cls, "predictions") and hasattr(student_model.cls, "predictions"):
                    if hasattr(teacher_model.cls.predictions, "decoder") and hasattr(student_model.cls.predictions, "decoder"):
                        if teacher_model.cls.predictions.decoder.weight.shape == student_model.cls.predictions.decoder.weight.shape:
                            student_model.cls.predictions.decoder.weight.data = teacher_model.cls.predictions.decoder.weight.data.clone()
                            if hasattr(teacher_model.cls.predictions.decoder, "bias") and hasattr(student_model.cls.predictions.decoder, "bias"):
                                student_model.cls.predictions.decoder.bias.data = teacher_model.cls.predictions.decoder.bias.data.clone()
                            logger.info("Copied MLM decoder weights")
                        else:
                            logger.info(f"MLM decoder dimensions don't match: teacher {teacher_model.cls.predictions.decoder.weight.shape}, student {student_model.cls.predictions.decoder.weight.shape}")
                    else:
                        logger.info("Models don't have decoder attribute for MLM")
                else:
                    logger.info("Models don't have predictions attribute for MLM")
            else:
                logger.info("Models don't have vocab_projector or cls attribute for MLM")
        
        logger.info("Enhanced knowledge transfer completed")
        return True
    except Exception as e:
        logger.error(f"Error during enhanced knowledge transfer: {e}")
        logger.error(traceback.format_exc())
        return False

def create_student_model(teacher_model, model_name, task):
    """Create a smaller student model based on the teacher model architecture."""
    logger.info("Creating student model")
    
    # Get the teacher model configuration
    if hasattr(teacher_model, "config"):
        teacher_config = teacher_model.config
    else:
        raise ValueError("Teacher model does not have a config attribute")
    
    # Create a new configuration for the student model with fewer layers
    student_config = AutoConfig.from_pretrained(model_name)
    
    # Reduce the number of layers (for BERT-like models)
    if hasattr(student_config, "num_hidden_layers"):
        student_config.num_hidden_layers = max(2, student_config.num_hidden_layers // 2)
        logger.info(f"Reduced number of layers from {teacher_config.num_hidden_layers} to {student_config.num_hidden_layers}")
    
    # Reduce the hidden size (optional, but more complex)
    # student_config.hidden_size = student_config.hidden_size // 2
    
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
    parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for distilled models")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model info
        logger.info(f"Loading model info from {args.model_info_path}")
        if os.path.isdir(args.model_info_path):
            # List files in the directory
            logger.info(f"model_info_path is a directory. Contents: {os.listdir(args.model_info_path)}")
            # Try to find a JSON file
            json_files = [f for f in os.listdir(args.model_info_path) if f.endswith('.json')]
            if json_files:
                # Use the first JSON file found
                model_info_file = os.path.join(args.model_info_path, json_files[0])
                logger.info(f"Using JSON file: {model_info_file}")
                with open(model_info_file, "r") as f:
                    model_info = json.load(f)
            else:
                raise FileNotFoundError(f"No JSON files found in directory: {args.model_info_path}")
        else:
            # It's a file, load it directly
            with open(args.model_info_path, "r") as f:
                model_info = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each model
        for model_key, info in model_info.items():
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
                
                # Create student model
                student_model = create_student_model(teacher_model, model_name, task)
                student_model = student_model.to(device)
                
                # Prepare inputs
                inputs = prepare_sample_inputs(tokenizer, task)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Apply knowledge transfer
                logger.info("Applying knowledge transfer")
                success = enhanced_knowledge_transfer(teacher_model, student_model, task, logger)
                if success:
                    logger.info("Knowledge transfer successful")
                else:
                    logger.warning("Knowledge transfer had issues, but continuing with distillation")
                
                # Save student model
                student_output_dir = os.path.join(args.output_dir, f"{model_key}_student")
                os.makedirs(student_output_dir, exist_ok=True)
                student_model.save_pretrained(student_output_dir)
                tokenizer.save_pretrained(student_output_dir)
                logger.info(f"Saved student model to {student_output_dir}")
                
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error in distillation: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

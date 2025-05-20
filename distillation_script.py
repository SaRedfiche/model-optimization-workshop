"""
Knowledge distillation script for model optimization workshop.
This script distills knowledge from a teacher model to a smaller student model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM, DistilBertConfig
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification
from transformers import DistilBertForQuestionAnswering, DistilBertForMaskedLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def load_teacher_model(model_name, task):
    """Load teacher model and tokenizer based on task."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if task == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task == "masked-lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return model, tokenizer

def create_student_model(teacher_model, task, student_model_name="distilbert-base-uncased"):
    """Create a student model based on the teacher model."""
    # Get teacher config
    teacher_config = teacher_model.config
    
    # Create student config
    if task == "sequence-classification":
        student_model = DistilBertForSequenceClassification.from_pretrained(
            student_model_name,
            num_labels=teacher_config.num_labels
        )
    elif task == "token-classification":
        student_model = DistilBertForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=teacher_config.num_labels
        )
    elif task == "question-answering":
        student_model = DistilBertForQuestionAnswering.from_pretrained(
            student_model_name
        )
    elif task == "masked-lm":
        student_model = DistilBertForMaskedLM.from_pretrained(
            student_model_name
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return student_model

def prepare_dataset(task):
    """Prepare a dataset for distillation based on the task."""
    if task == "sequence-classification":
        # Use SST-2 dataset for sentiment analysis
        dataset = load_dataset("glue", "sst2")
        return dataset["train"].select(range(1000))  # Use a subset for faster training
    elif task == "token-classification":
        # Use CoNLL-2003 dataset for NER
        dataset = load_dataset("conll2003")
        return dataset["train"].select(range(1000))  # Use a subset for faster training
    elif task == "question-answering":
        # Use SQuAD dataset for question answering
        dataset = load_dataset("squad")
        return dataset["train"].select(range(1000))  # Use a subset for faster training
    elif task == "masked-lm":
        # Use WikiText dataset for masked language modeling
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        return dataset["train"].select(range(1000))  # Use a subset for faster training
    else:
        raise ValueError(f"Unsupported task: {task}")

def tokenize_dataset(dataset, tokenizer, task):
    """Tokenize the dataset based on the task."""
    if task == "sequence-classification":
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    elif task == "token-classification":
        def tokenize_function(examples):
            return tokenizer(examples["tokens"], is_split_into_words=True, padding="max_length", truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    elif task == "question-answering":
        def tokenize_function(examples):
            return tokenizer(
                examples["question"],
                examples["context"],
                padding="max_length",
                truncation=True
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    elif task == "masked-lm":
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    else:
        raise ValueError(f"Unsupported task: {task}")

def distill_model(teacher_model, student_model, tokenized_dataset, task, output_dir, num_epochs=3, batch_size=8):
    """Distill knowledge from teacher to student model."""
    # Define distillation parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Create a custom trainer for distillation
    class DistillationTrainer(Trainer):
        def __init__(self, *args, teacher_model=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model
            
        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass through student model
            outputs = model(**inputs)
            student_logits = outputs.logits
            
            # Forward pass through teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Compute distillation loss (soft targets)
            temperature = 2.0
            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1)
            ) * (temperature ** 2)
            
            # Return loss
            return (distillation_loss, outputs) if return_outputs else distillation_loss
    
    # Create trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        teacher_model=teacher_model
    )
    
    # Train the student model
    trainer.train()
    
    # Save the student model
    student_model.save_pretrained(output_dir)
    
    return student_model

def measure_inference_time(model, inputs, num_runs=10):
    """Measure inference time for a model."""
    # Warm-up run
    with torch.no_grad():
        _ = model(**inputs)
    
    # Measure inference time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time.record()
            _ = model(**inputs)
            end_time.record()
            torch.cuda.synchronize()
            timings.append(start_time.elapsed_time(end_time))
    
    return sum(timings) / len(timings)

def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def measure_memory_usage(model):
    """Measure memory usage of a model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.to('cuda')
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2
        model.to('cpu')
        return memory_usage
    else:
        return get_model_size(model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--student-model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    
    # Load model info
    with open(args.model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Process each model
    distilled_metrics = {}
    for model_key, info in model_info.items():
        print(f"Processing {model_key}: {info['model_name']}")
        
        # Skip models that are not suitable for distillation
        if info['task'] not in ["sequence-classification", "token-classification", "question-answering", "masked-lm"]:
            print(f"Skipping {model_key}: task {info['task']} not supported for distillation")
            continue
        
        # Load teacher model and tokenizer
        teacher_model, tokenizer = load_teacher_model(info['model_name'], info['task'])
        
        # Create student model
        student_model = create_student_model(teacher_model, info['task'], args.student_model_name)
        
        # Prepare dataset
        dataset = prepare_dataset(info['task'])
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, info['task'])
        
        # Distill knowledge from teacher to student
        model_output_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(model_output_dir, exist_ok=True)
        
        student_model = distill_model(
            teacher_model,
            student_model,
            tokenized_dataset,
            info['task'],
            model_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(model_output_dir)
        
        # Define sample input
        if info['task'] == 'sequence-classification':
            sample_input = "This is a sample input for sentiment analysis."
            inputs = tokenizer(sample_input, return_tensors="pt")
        elif info['task'] == 'token-classification':
            sample_input = "John Smith works at Microsoft in Seattle."
            inputs = tokenizer(sample_input, return_tensors="pt")
        elif info['task'] == 'question-answering':
            sample_input = {
                "question": "What is machine learning?",
                "context": "Machine learning is a branch of artificial intelligence."
            }
            inputs = tokenizer(
                sample_input["question"],
                sample_input["context"],
                return_tensors="pt"
            )
        elif info['task'] == 'masked-lm':
            sample_input = "The [MASK] is a large language model."
            inputs = tokenizer(sample_input, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            teacher_model = teacher_model.to('cuda')
            student_model = student_model.to('cuda')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Measure metrics
        teacher_size = get_model_size(teacher_model)
        student_size = get_model_size(student_model)
        teacher_inference_time = measure_inference_time(teacher_model, inputs)
        student_inference_time = measure_inference_time(student_model, inputs)
        teacher_memory_usage = measure_memory_usage(teacher_model)
        student_memory_usage = measure_memory_usage(student_model)
        teacher_parameters = sum(p.numel() for p in teacher_model.parameters())
        student_parameters = sum(p.numel() for p in student_model.parameters())
        
        # Save metrics
        distilled_metrics[f"{model_key}_distilled"] = {
            "model_key": f"{model_key}_distilled",
            "model_name": f"distilled_{info['model_name']}",
            "task": info['task'],
            "teacher_model_key": model_key,
            "student_model_name": args.student_model_name,
            "model_size": student_size,
            "inference_time": student_inference_time,
            "memory_usage": student_memory_usage,
            "num_parameters": student_parameters,
            "teacher_model_size": teacher_size,
            "teacher_inference_time": teacher_inference_time,
            "teacher_memory_usage": teacher_memory_usage,
            "teacher_num_parameters": teacher_parameters,
            "size_reduction": (teacher_size - student_size) / teacher_size * 100,
            "speedup": teacher_inference_time / student_inference_time
        }
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'distilled_metrics.json'), 'w') as f:
        json.dump(distilled_metrics, f, indent=2)

if __name__ == '__main__':
    main()

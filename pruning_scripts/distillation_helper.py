#!/usr/bin/env python3
"""
Helper module for knowledge distillation after pruning.
This module provides functions to distill knowledge from a pruned model to a smaller student model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Simple dataset for text samples."""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension added by tokenizer
        return {k: v.squeeze(0) for k, v in encoding.items()}

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        temperature: Temperature for softening the distributions
        
    Returns:
        Distillation loss
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss
    loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    return loss

def distill_from_pruned_model(teacher_model, student_model, dataset_loader, optimizer, device, 
                             num_epochs=3, temperature=2.0):
    """
    Distill knowledge from a pruned teacher model to a smaller student model.
    
    Args:
        teacher_model: The pruned teacher model
        student_model: The smaller student model
        dataset_loader: DataLoader with training examples
        optimizer: Optimizer for the student model
        device: Device to run training on
        num_epochs: Number of training epochs
        temperature: Temperature for knowledge distillation
        
    Returns:
        Trained student model
    """
    teacher_model.to(device)
    student_model.to(device)
    
    teacher_model.eval()  # Teacher model is always in eval mode
    
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0
        
        for batch in dataset_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with teacher model (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits
            
            # Forward pass with student model
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits
            
            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_logits, temperature)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return student_model

def create_smaller_model(teacher_model, model_name, num_labels):
    """
    Create a smaller version of the teacher model for distillation.
    
    Args:
        teacher_model: The teacher model to get configuration from
        model_name: Base model name for the smaller model
        num_labels: Number of output labels
        
    Returns:
        Smaller student model
    """
    # For this example, we'll use a smaller pre-trained model
    # In a real scenario, you might want to create a custom smaller architecture
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return student_model

def export_distilled_model(model, tokenizer, output_path):
    """
    Export the distilled model for deployment.
    
    Args:
        model: The distilled model
        tokenizer: The tokenizer
        output_path: Path to save the model
        
    Returns:
        Path to the saved model
    """
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

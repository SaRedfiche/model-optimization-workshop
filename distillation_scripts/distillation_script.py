#!/usr/bin/env python3
"""
Simplified Knowledge Distillation script for SageMaker Processing Jobs.
This version focuses on core functionality and robust error handling.
"""

# Force CPU usage at the very beginning to avoid MPS issues
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import sys
import json
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function with comprehensive error handling."""
    try:
        logger.info("=== Starting Knowledge Distillation Process ===")
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='Perform knowledge distillation')
        parser.add_argument('--teacher-info-path', type=str, required=True)
        parser.add_argument('--student-info-path', type=str, required=True)
        parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
        parser.add_argument('--temperature', type=float, default=2.0)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=8)
        
        args = parser.parse_args()
        
        logger.info(f"Arguments: {vars(args)}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")
        
        # Check input files
        if not os.path.exists(args.teacher_info_path):
            raise FileNotFoundError(f"Teacher info file not found: {args.teacher_info_path}")
        if not os.path.exists(args.student_info_path):
            raise FileNotFoundError(f"Student info file not found: {args.student_info_path}")
        
        logger.info("Input files validated successfully")
        
        # Load configurations
        with open(args.teacher_info_path, 'r') as f:
            teacher_info = json.load(f)
        with open(args.student_info_path, 'r') as f:
            student_info = json.load(f)
        
        logger.info(f"Loaded teacher info: {list(teacher_info.keys())}")
        logger.info(f"Loaded student info: {list(student_info.keys())}")
        
        # Install dependencies
        logger.info("Installing dependencies...")
        os.system("pip install --quiet torch transformers datasets accelerate")
        
        # Import after installation
        import torch
        import torch.nn.functional as F
        from transformers import (
            AutoModelForSequenceClassification, 
            AutoTokenizer, 
            AutoConfig,
            DistilBertConfig,
            Trainer, 
            TrainingArguments,
            DataCollatorWithPadding
        )
        from datasets import Dataset
        
        # Completely disable MPS backend
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False
            torch.backends.mps.is_built = lambda: False
        
        logger.info("Dependencies imported successfully")
        
        # Force CPU usage
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Get model info
        teacher_key = list(teacher_info.keys())[0]
        student_key = list(student_info.keys())[0]
        teacher_data = teacher_info[teacher_key]
        student_config = student_info[student_key]
        
        logger.info(f"Teacher model: {teacher_data['model_name']}")
        logger.info(f"Student config: {student_config}")
        
        # Load teacher model
        logger.info("Loading teacher model...")
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            teacher_data['model_name']
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(teacher_data['model_name'])
        
        # Create student model
        logger.info("Creating student model...")
        teacher_config = teacher_model.config
        
        config = DistilBertConfig(
            vocab_size=teacher_config.vocab_size,
            max_position_embeddings=512,
            sinusoidal_pos_embds=False,
            n_layers=student_config.get('num_hidden_layers', 3),
            n_heads=8,
            dim=student_config.get('hidden_size', 384),
            hidden_dim=student_config.get('hidden_size', 384) * 4,
            dropout=0.1,
            attention_dropout=0.1,
            activation='gelu',
            initializer_range=0.02,
            qa_dropout=0.1,
            seq_classif_dropout=0.2,
            num_labels=teacher_config.num_labels,
            id2label=teacher_config.id2label,
            label2id=teacher_config.label2id
        )
        
        student_model = AutoModelForSequenceClassification.from_config(config).to(device)
        
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        logger.info(f"Teacher parameters: {teacher_params:,}")
        logger.info(f"Student parameters: {student_params:,}")
        
        # Create simple dataset
        logger.info("Creating training dataset...")
        texts = [
            "I love this movie, it's fantastic!",
            "This is the worst film I've ever seen.",
            "The movie was okay, nothing special.",
            "Amazing acting and great storyline!",
            "I didn't like it at all.",
            "Pretty good movie, would recommend.",
        ] * 50  # Small dataset for quick training
        
        labels = [1, 0, 1, 1, 0, 1] * 50
        
        dataset = Dataset.from_dict({'text': texts, 'labels': labels})
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Custom trainer
        class SimpleDistillationTrainer(Trainer):
            def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.temperature = temperature
                self.alpha = alpha
                self.teacher_model.eval()
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                student_outputs = model(**inputs)
                student_logits = student_outputs.logits
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Distillation loss
                distillation_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
                
                # Task loss
                task_loss = F.cross_entropy(student_logits, inputs['labels'])
                
                # Combined loss
                loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
                
                return (loss, student_outputs) if return_outputs else loss
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "temp"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            logging_steps=10,
            save_steps=1000,
            eval_strategy="no",
            save_strategy="no",
            report_to=None,
            use_mps_device=False,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = SimpleDistillationTrainer(
            teacher_model=teacher_model,
            temperature=args.temperature,
            alpha=args.alpha,
            model=student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        student_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save info
        size_reduction = (teacher_params - student_params) / teacher_params * 100
        
        model_info = {
            "teacher_model": teacher_data['model_name'],
            "student_config": student_config,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "size_reduction_percent": size_reduction,
            "distilled_model_path": args.output_dir
        }
        
        with open(os.path.join(args.output_dir, "distillation_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"=== Distillation completed successfully! ===")
        logger.info(f"Size reduction: {size_reduction:.2f}%")
        logger.info(f"Output saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"=== DISTILLATION FAILED ===")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

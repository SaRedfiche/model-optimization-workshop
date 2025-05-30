def train_student_model(teacher_model, student_model, tokenizer, task, device, args, logger):
    """Train the student model using knowledge distillation from the teacher model."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os
    import traceback
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    
    try:
        logger.info("Setting up training dataset")
        
        # Load appropriate dataset based on task
        if task == "sequence-classification" or task == "text-classification":
            logger.info("Loading SST-2 dataset for text classification")
            dataset = load_dataset("glue", "sst2")
            train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))  # Limit size for speed
            eval_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
            
            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
            
            # Define accuracy function
            def compute_accuracy(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {"accuracy": (predictions == labels).mean()}
            
        elif task == "token-classification":
            logger.info("Loading CoNLL-2003 dataset for token classification")
            dataset = load_dataset("conll2003")
            train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
            eval_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
            
            # Tokenize dataset
            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
                labels = []
                for i, label in enumerate(examples["ner_tags"]):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    labels.append(label_ids)
                tokenized_inputs["labels"] = labels
                return tokenized_inputs
            
            tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
            tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True)
            
            # Define accuracy function
            def compute_accuracy(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=2)
                
                # Remove ignored index (special tokens)
                true_predictions = [
                    [p for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(predictions, labels)
                ]
                true_labels = [
                    [l for l in label if l != -100]
                    for label in labels
                ]
                
                # Calculate accuracy
                correct = sum(pred == label for preds, labels in zip(true_predictions, true_labels) 
                             for pred, label in zip(preds, labels))
                total = sum(len(labels) for labels in true_labels)
                return {"accuracy": correct / total if total > 0 else 0}
                
        elif task == "question-answering":
            logger.info("Loading SQuAD dataset for question answering")
            dataset = load_dataset("squad")
            train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
            eval_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
            
            # Tokenize dataset
            def prepare_qa_features(examples):
                questions = [q.strip() for q in examples["question"]]
                contexts = [c.strip() for c in examples["context"]]
                
                # Tokenize
                tokenized = tokenizer(
                    questions,
                    contexts,
                    truncation="only_second",
                    max_length=384,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
                
                # Map original indices
                sample_map = tokenized.pop("overflow_to_sample_mapping")
                offset_mapping = tokenized.pop("offset_mapping")
                
                # Get start and end positions
                tokenized["start_positions"] = []
                tokenized["end_positions"] = []
                
                for i, offsets in enumerate(offset_mapping):
                    input_ids = tokenized["input_ids"][i]
                    cls_index = input_ids.index(tokenizer.cls_token_id)
                    
                    # Get sample index
                    sample_idx = sample_map[i]
                    
                    # Get answer
                    answer = examples["answers"][sample_idx]
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])
                    
                    # Find token positions
                    start_position = cls_index
                    end_position = cls_index
                    
                    for j, (start, end) in enumerate(offsets):
                        if start <= start_char < end:
                            start_position = j
                        if start < end_char <= end:
                            end_position = j
                    
                    tokenized["start_positions"].append(start_position)
                    tokenized["end_positions"].append(end_position)
                
                return tokenized
            
            tokenized_train = train_dataset.map(prepare_qa_features, batched=True, remove_columns=train_dataset.column_names)
            tokenized_eval = eval_dataset.map(prepare_qa_features, batched=True, remove_columns=eval_dataset.column_names)
            
            # Define accuracy function (simplified for QA)
            def compute_accuracy(eval_pred):
                # For QA, we'll use a simplified exact match metric
                predictions, labels = eval_pred
                start_preds = np.argmax(predictions[0], axis=1)
                end_preds = np.argmax(predictions[1], axis=1)
                start_labels = labels[0]
                end_labels = labels[1]
                
                # Calculate exact match
                exact_match = ((start_preds == start_labels) & (end_preds == end_labels)).mean()
                return {"accuracy": exact_match}
                
        elif task == "masked-lm" or task == "fill-mask":
            logger.info("Loading WikiText dataset for masked language modeling")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
            eval_dataset = dataset["test"].select(range(min(1000, len(dataset["test"]))))
            
            # Tokenize dataset
            def tokenize_function(examples):
                result = tokenizer(examples["text"])
                if tokenizer.is_fast:
                    result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
                return result
                
            tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            
            # Group texts for MLM
            def group_texts(examples):
                block_size = 128
                concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated["input_ids"])
                total_length = (total_length // block_size) * block_size
                
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated.items()
                }
                
                result["labels"] = result["input_ids"].copy()
                return result
                
            tokenized_train = tokenized_train.map(group_texts, batched=True)
            tokenized_eval = tokenized_eval.map(group_texts, batched=True)
            
            # Define accuracy function for MLM (using perplexity)
            def compute_accuracy(eval_pred):
                predictions, labels = eval_pred
                # Mask out padding tokens
                mask = labels != -100
                
                # Calculate accuracy only on masked tokens
                predictions = np.argmax(predictions, axis=2)
                correct = (predictions[mask] == labels[mask]).sum()
                total = mask.sum()
                return {"accuracy": correct / total if total > 0 else 0}
        
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        # Define distillation trainer
        class DistillationTrainer(Trainer):
            def __init__(self, *args, teacher_model=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.temperature = args[1].temperature  # Access temperature from training_args
                self.alpha = args[1].alpha  # Access alpha from training_args
                
            def compute_loss(self, model, inputs, return_outputs=False):
                # Standard student loss
                outputs = model(**inputs)
                student_loss = outputs.loss
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                
                # Distillation loss
                if task == "sequence-classification" or task == "text-classification":
                    student_logits = outputs.logits
                    teacher_logits = teacher_outputs.logits
                    
                    # Apply temperature scaling
                    student_logits_t = student_logits / self.temperature
                    teacher_logits_t = teacher_logits / self.temperature
                    
                    # KL divergence loss
                    distillation_loss = F.kl_div(
                        F.log_softmax(student_logits_t, dim=-1),
                        F.softmax(teacher_logits_t, dim=-1),
                        reduction="batchmean",
                    ) * (self.temperature ** 2)
                    
                    # Combined loss
                    loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                else:
                    # For other tasks, just use the student loss for now
                    loss = student_loss
                
                return (loss, outputs) if return_outputs else loss
        
        # Create trainer
        trainer = DistillationTrainer(
            model=student_model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_accuracy,
            teacher_model=teacher_model,
        )
        
        # Train the model
        logger.info("Starting student model training")
        trainer.train()
        
        # Evaluate the model
        logger.info("Evaluating student model")
        eval_results = trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_results}")
        
        # Return accuracy
        return eval_results.get("eval_accuracy", 0)
        
    except Exception as e:
        logger.error(f"Error training student model: {e}")
        logger.error(traceback.format_exc())
        return 0

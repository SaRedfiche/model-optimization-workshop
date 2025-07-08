# Import necessary libraries for inference
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertConfig
from transformers import pipeline
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample text for sentiment analysis
sample_texts = [
    "I really enjoyed this movie. The acting was superb and the plot was engaging.",
    "This product is terrible. It broke after just one use and customer service was unhelpful.",
    "The restaurant was okay. Food was good but the service was slow."
]

print("Demonstrating inference with a default sentiment analysis model...")

try:
    # Use a default model for demonstration
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"\nLoading default model: {model_name}")
    
    # Load the teacher model and tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create a pipeline for the teacher model
    teacher_pipeline = pipeline("sentiment-analysis", model=teacher_model, tokenizer=teacher_tokenizer)
    
    # Create a smaller student model
    print("\nCreating a representative student model for demonstration...")
    
    # Get the number of labels and label mappings from the teacher model
    num_labels = teacher_model.config.num_labels
    id2label = teacher_model.config.id2label
    label2id = teacher_model.config.label2id
    
    print(f"Teacher model label mapping: {id2label}")
    
    # Create a proper DistilBert configuration with fewer layers
    student_config = DistilBertConfig(
        vocab_size=teacher_model.config.vocab_size,
        max_position_embeddings=teacher_model.config.max_position_embeddings,
        sinusoidal_pos_embds=False,
        n_layers=3,  # Fewer layers
        n_heads=8,   # Same number of attention heads
        dim=384,     # Smaller hidden size
        hidden_dim=1536,  # Smaller intermediate size
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        num_labels=num_labels,  # Same number of output labels
        id2label=id2label,      # Copy label mapping from teacher
        label2id=label2id       # Copy label mapping from teacher
    )
    
    # Create a smaller student model
    student_model = AutoModelForSequenceClassification.from_config(student_config)
    
    # Create a pipeline for the student model
    student_pipeline = pipeline("sentiment-analysis", model=student_model, tokenizer=teacher_tokenizer)
    
    # Compare inference results
    print("\nComparing inference results:")
    print("\nTeacher model results:")
    teacher_results = teacher_pipeline(sample_texts)
    for text, result in zip(sample_texts, teacher_results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")
        print()
    
    print("\nStudent model results (representative):")
    student_results = student_pipeline(sample_texts)
    for text, result in zip(sample_texts, student_results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")
        print()
    
    # Function to measure inference time
    def measure_inference_time(model, tokenizer, text, num_runs=10):
        # Create inputs
        inputs = tokenizer(text, return_tensors="pt")
        
        # Warm-up run
        with torch.no_grad():
            model(**inputs)
        
        # Measure inference time
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return sum(inference_times) / len(inference_times), outputs
    
    # Measure and compare inference time
    print("\nMeasuring inference time...")
    teacher_time, _ = measure_inference_time(teacher_model, teacher_tokenizer, sample_texts[0])
    student_time, _ = measure_inference_time(student_model, teacher_tokenizer, sample_texts[0])
    
    print(f"Teacher model inference time: {teacher_time:.2f} ms")
    print(f"Student model inference time: {student_time:.2f} ms")
    print(f"Speed improvement: {(teacher_time - student_time) / teacher_time * 100:.2f}%")
    
    # Compare model sizes
    teacher_size = sum(p.numel() for p in teacher_model.parameters())
    student_size = sum(p.numel() for p in student_model.parameters())
    
    print(f"\nTeacher model parameters: {teacher_size:,}")
    print(f"Student model parameters: {student_size:,}")
    print(f"Size reduction: {(teacher_size - student_size) / teacher_size * 100:.2f}%")
    
    # Visualize the comparison
    comparison_data = {
        'Model': ['Teacher', 'Student'],
        'Inference Time (ms)': [teacher_time, student_time],
        'Parameters': [teacher_size, student_size]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot inference time comparison
    sns.barplot(x='Model', y='Inference Time (ms)', data=df, ax=ax1, palette='viridis')
    ax1.set_title('Inference Time Comparison')
    
    # Plot parameter count comparison
    sns.barplot(x='Model', y='Parameters', data=df, ax=ax2, palette='viridis')
    ax2.set_title('Parameter Count Comparison')
    ax2.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\nNote: This is a demonstration with a default model.")
    print("In a real deployment, you would use the actual distilled model from S3.")
    print("The actual distilled model would have similar size characteristics but better accuracy")
    print("due to the knowledge transfer from the teacher model during distillation.")
except Exception as e:
    print(f"Error demonstrating inference with default model: {e}")
    import traceback
    traceback.print_exc()

# Model Pruning Techniques for LLMs

This document explores pruning techniques for large language models (LLMs) like Llama, BERT, GPT, and DeepSeek to reduce their footprint for more cost-effective hosting on AWS SageMaker.

## Pruning Overview

Pruning reduces model size by removing unnecessary weights, neurons, or entire components from neural networks. For LLMs, pruning can significantly reduce model size with minimal impact on performance when done correctly.

## Types of Pruning

### Unstructured Pruning

**Description**: Removes individual weights based on importance criteria (typically magnitude).

**Advantages**:
- Higher theoretical compression rates
- More fine-grained control over pruning

**Disadvantages**:
- Requires specialized hardware/software for speed benefits
- Irregular sparsity patterns

**Compression Rate**: Up to 80-90% of weights can be pruned (10-20% remaining)

**Implementation**:
```python
import torch
from torch.nn.utils import prune

# Example for a single layer
module = model.transformer.h[0].mlp.dense_4h_to_h
prune.l1_unstructured(module, name="weight", amount=0.3)  # Prune 30% of weights

# Make pruning permanent
prune.remove(module, "weight")
```

### Structured Pruning

**Description**: Removes entire structures (neurons, attention heads, or layers).

**Advantages**:
- Works on standard hardware without specialized libraries
- Directly reduces computation and memory requirements

**Disadvantages**:
- Generally lower compression rates than unstructured pruning
- May require more careful selection of what to prune

**Compression Rate**: Typically 30-50% of structures can be pruned

**Implementation**:
```python
# Example: Pruning attention heads in a transformer model
# Assuming model has a mask for attention heads
model.set_attention_mask([
    [1, 1, 0, 1],  # Keep heads 0, 1, 3 in layer 0
    [1, 0, 1, 0],  # Keep heads 0, 2 in layer 1
    # ... and so on
])
```

## LLM-Specific Pruning Techniques

### Attention Head Pruning

**Description**: Removes less important attention heads from transformer models.

**Method**:
1. Compute importance scores for each attention head
2. Remove heads with lowest scores
3. Fine-tune the pruned model to recover performance

**Effectiveness**: 
- BERT: Up to 30-40% of heads can be pruned with <1% accuracy drop
- GPT models: 20-30% of heads with minimal perplexity increase

**Implementation**:
```python
# Pseudo-code for attention head pruning
importance_scores = []
for layer in model.layers:
    for head in layer.attention.heads:
        # Compute importance score (e.g., using gradient-based methods)
        score = compute_head_importance(model, head, evaluation_data)
        importance_scores.append((layer, head, score))

# Sort by importance and prune least important heads
importance_scores.sort(key=lambda x: x[2])
for layer, head, score in importance_scores[:num_heads_to_prune]:
    prune_attention_head(model, layer, head)
```

### Layer Pruning

**Description**: Removes entire transformer layers from the model.

**Method**:
1. Identify less important layers using contribution metrics
2. Remove these layers and reconnect the network
3. Fine-tune to recover performance

**Effectiveness**:
- BERT: Up to 40-50% of layers can be pruned with proper fine-tuning
- GPT/Llama: More sensitive, typically 20-30% of layers

**Implementation**:
```python
# Example: Layer dropping in Hugging Face transformers
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Identify layers to keep (e.g., every other layer)
layers_to_keep = list(range(0, model.config.num_hidden_layers, 2))

# Create pruned config
pruned_config = model.config
pruned_config.num_hidden_layers = len(layers_to_keep)

# Create new model with fewer layers
pruned_model = AutoModelForCausalLM.from_config(pruned_config)

# Copy weights from original model to pruned model
for i, j in enumerate(layers_to_keep):
    pruned_model.transformer.h[i].load_state_dict(
        model.transformer.h[j].state_dict()
    )
```

### Movement Pruning

**Description**: Specialized technique that removes weights that move toward zero during fine-tuning.

**Method**:
1. Add regularization that pushes unimportant weights to zero
2. During fine-tuning, gradually prune weights that approach zero
3. Continue fine-tuning the sparse model

**Effectiveness**: Up to 70-80% sparsity with minimal performance impact

**Implementation**:
```python
# Using Hugging Face's PEFT library for movement pruning
from transformers import AutoModelForCausalLM
from peft import MovementPruningConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Configure movement pruning
peft_config = MovementPruningConfig(
    target_modules=["query", "key", "value", "dense"],
    sparsity_target=0.7,  # Target 70% sparsity
)

# Create prunable model
prunable_model = get_peft_model(model, peft_config)

# Fine-tune with movement pruning
# (training code here)
```

## Pruning + Quantization Combination

**Description**: Applying both pruning and quantization for multiplicative benefits.

**Method**:
1. Prune the model first to remove unnecessary weights
2. Quantize the remaining weights to lower precision
3. Fine-tune if necessary to recover performance

**Effectiveness**: Combined 10-20x reduction possible

**Implementation**:
```python
# Pseudo-code for pruning + quantization
# 1. Prune
pruned_model = apply_pruning(model, sparsity=0.5)  # 50% sparsity

# 2. Fine-tune pruned model
fine_tuned_model = fine_tune(pruned_model, training_data)

# 3. Quantize the pruned model
quantized_model = quantize_to_int8(fine_tuned_model)
```

## SageMaker Implementation

### Custom Training Script for Pruning
```python
# sagemaker_pruning_script.py
import argparse
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from torch.nn.utils import prune

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b")
parser.add_argument("--pruning-rate", type=float, default=0.3)
args, _ = parser.parse_known_args()

# Load model
model = AutoModelForCausalLM.from_pretrained(args.model_id)

# Apply pruning to linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=args.pruning_rate)

# Fine-tune pruned model
training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset(),  # Load your dataset
)

trainer.train()

# Save pruned model
model.save_pretrained("/opt/ml/model")
```

### SageMaker Training Job
```python
import sagemaker
from sagemaker.huggingface import HuggingFace

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point="sagemaker_pruning_script.py",
    source_dir="./scripts",
    role=role,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py39",
    hyperparameters={
        "model-id": "meta-llama/Llama-2-7b",
        "pruning-rate": 0.3,
    },
)

# Start training job
huggingface_estimator.fit()
```

## Cost Analysis

### Pruning Process Costs
- **Layer Pruning**:
  - Hardware: ml.g5.2xlarge ($1.515/hour)
  - Time: 1-2 hours for pruning + 12-24 hours for fine-tuning
  - Total: $20-$40
  
- **Unstructured Pruning**:
  - Hardware: ml.g5.2xlarge ($1.515/hour)
  - Time: 2-3 hours for pruning + 24-48 hours for fine-tuning
  - Total: $40-$80

### Hosting Cost Comparison
- **Original 7B Model**:
  - Instance: ml.g5.2xlarge ($1.515/hour)
  - Monthly cost (24/7): ~$1,100
  
- **Pruned 7B Model (50% reduction)**:
  - Instance: ml.g5.xlarge ($0.7575/hour)
  - Monthly cost (24/7): ~$550
  - Savings: ~50%

- **Pruned + Quantized Model**:
  - Instance: ml.g5.xlarge or smaller
  - Monthly cost (24/7): ~$400-$550
  - Savings: ~50-60%

## Performance Impact

| Model | Pruning Method | Reduction | Performance Impact | Fine-tuning Required |
|-------|---------------|-----------|-------------------|---------------------|
| BERT-Large | Head Pruning (30%) | ~15% | <1% accuracy drop | Minimal |
| BERT-Large | Layer Pruning (40%) | ~40% | 1-2% accuracy drop | Yes |
| Llama-2-7B | Unstructured (50%) | ~50% | 2-3% perplexity increase | Yes |
| Llama-2-7B | Layer Pruning (25%) | ~25% | 1-2% perplexity increase | Yes |
| GPT-J-6B | Movement Pruning (70%) | ~70% | 2-4% perplexity increase | Yes |

## Conclusion

Pruning offers significant size reduction for LLMs but typically requires fine-tuning to maintain performance. The best approach depends on:

1. **Deployment constraints**: If specialized hardware for sparse computation is available, unstructured pruning offers better compression rates.

2. **Performance requirements**: More aggressive pruning leads to greater performance degradation.

3. **Fine-tuning resources**: Layer pruning requires less fine-tuning to recover performance compared to unstructured pruning.

4. **Combination with other techniques**: Pruning works well when combined with quantization for multiplicative benefits.

For AWS SageMaker deployment, structured pruning (especially layer pruning) offers the most straightforward path, as it doesn't require specialized libraries for inference speed benefits.

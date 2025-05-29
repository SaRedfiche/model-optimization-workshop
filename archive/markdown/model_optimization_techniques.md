# Model Optimization Techniques for AWS SageMaker

This document tracks research on techniques to reduce ML model footprint for more cost-effective hosting on AWS SageMaker, focusing on large language models (LLMs) like Llama, BERT, GPT, and DeepSeek with 8B+ parameters.

## Table of Contents
1. [Quantization Techniques](#quantization-techniques)
2. [Model Pruning](#model-pruning)
3. [Knowledge Distillation](#knowledge-distillation)
4. [Model Compression](#model-compression)
5. [Efficient Model Architectures](#efficient-model-architectures)
6. [SageMaker-Specific Optimizations](#sagemaker-specific-optimizations)
7. [Cost Analysis](#cost-analysis)
8. [Implementation Examples](#implementation-examples)

## Quantization Techniques

Quantization reduces the precision of the weights in a neural network, typically from 32-bit floating point to lower precision formats.

### Research Notes:
#### Post-Training Quantization (PTQ)
- **Description**: Converts model weights to lower precision after training is complete
- **Precision Options**:
  - FP16 (16-bit floating point): ~2x reduction with minimal accuracy loss
  - INT8 (8-bit integer): ~4x reduction with some accuracy impact
  - INT4 (4-bit integer): ~8x reduction with more significant accuracy impact
- **Framework Support**:
  - **PyTorch**: 
    - `torch.quantization` API
    - NVIDIA TensorRT integration
    - Hugging Face Optimum library
  - **TensorFlow**: 
    - TF-Lite quantization
    - TensorFlow Model Optimization Toolkit

#### Quantization-Aware Training (QAT)
- **Description**: Simulates quantization effects during training to minimize accuracy loss
- **Benefits**: Better accuracy preservation compared to PTQ
- **Frameworks**:
  - **PyTorch**: `torch.quantization.quantize_qat`
  - **TensorFlow**: `tf.keras.quantization`

#### LLM-Specific Quantization
- **GPTQ**: Specialized for LLMs, achieves INT4 with minimal performance loss
- **AWQ (Activation-aware Weight Quantization)**: Optimizes for LLM inference
- **SmoothQuant**: Balances activation and weight quantization for LLMs
- **QLoRA**: Combines quantization with parameter-efficient fine-tuning

#### SageMaker Implementation
- SageMaker Neo supports automatic quantization
- SageMaker Inference Recommender can benchmark quantized models
- Custom containers can be used for specialized quantization techniques

## Model Pruning

Pruning removes unnecessary connections or neurons from a neural network.

### Research Notes:
#### Structured Pruning
- **Description**: Removes entire channels, neurons, or attention heads
- **LLM-Specific Techniques**:
  - **Attention Head Pruning**: Removes less important attention heads
  - **Layer Pruning**: Removes entire transformer layers
- **Frameworks**:
  - **PyTorch**: `torch.nn.utils.prune`
  - **TensorFlow**: TensorFlow Model Optimization Toolkit

#### Unstructured Pruning
- **Description**: Removes individual weights based on magnitude or importance
- **Techniques**:
  - **Magnitude Pruning**: Removes smallest weights
  - **Movement Pruning**: Removes weights that move toward zero during fine-tuning
- **LLM Considerations**: 
  - Requires specialized sparse computation libraries for speed benefits
  - Often combined with quantization

#### SageMaker Implementation
- Custom training scripts required
- Sparse computation libraries may need custom containers

## Knowledge Distillation

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model.

### Research Notes:
#### Standard Distillation
- **Description**: Student model trained on teacher model outputs
- **LLM Examples**:
  - DistilBERT (66M params) from BERT (110M params)
  - DistilGPT2 from GPT2
  - TinyLlama (1.1B params) from Llama (7B params)

#### Response-Based Distillation
- **Description**: Student learns from final outputs of teacher
- **Implementation**:
  - Loss = α * CE(student, labels) + (1-α) * CE(student, teacher_softmax)
  - Temperature parameter controls softness of teacher distribution

#### Feature-Based Distillation
- **Description**: Student learns intermediate representations from teacher
- **LLM Approach**: Match hidden states, attention patterns

#### SageMaker Implementation
- Standard training workflows with custom loss functions
- Can use SageMaker Training with spot instances to reduce costs

## Model Compression

Compression techniques reduce the size of model weights and architecture.

### Research Notes:
#### Weight Sharing
- **Description**: Multiple connections share the same weight value
- **Techniques**:
  - **Scalar Quantization**: Maps weights to a codebook
  - **Product Quantization**: Decomposes weight matrices into smaller subspaces

#### Low-Rank Factorization
- **Description**: Approximates weight matrices as product of smaller matrices
- **LLM Applications**:
  - LoRA (Low-Rank Adaptation): Adds low-rank updates during fine-tuning
  - PEFT (Parameter-Efficient Fine-Tuning): Various techniques including LoRA

#### Pruning + Quantization Combinations
- **Description**: Apply pruning first, then quantize the remaining weights
- **Benefits**: Multiplicative size reduction (e.g., 2x from pruning × 4x from INT8 = 8x total)

#### SageMaker Implementation
- Hugging Face Transformers integration with SageMaker
- Custom containers for specialized techniques

## Efficient Model Architectures

Designing or selecting model architectures that are inherently more efficient.

### Research Notes:
#### Smaller Base Models
- **Examples**:
  - Llama 2 7B vs. 13B vs. 70B
  - BERT-Base (110M) vs. BERT-Large (340M)
  - DeepSeek-V2 1.3B vs. 7B vs. 16B

#### Architectural Optimizations
- **Flash Attention**: Optimized attention computation
- **Multi-Query Attention**: Reduces key/value projections
- **Grouped-Query Attention (GQA)**: Balance between MHA and MQA

#### Mixture of Experts (MoE)
- **Description**: Routes tokens through specialized sub-networks
- **Examples**: 
  - DeepSeek-MoE
  - Mixtral 8x7B
- **Benefits**: Can maintain quality while reducing active parameters

#### SageMaker Implementation
- Pre-trained smaller models available through SageMaker JumpStart
- Custom training for specialized architectures

## SageMaker-Specific Optimizations

Optimizations specific to AWS SageMaker deployment.

### Research Notes:
#### SageMaker Neo
- **Description**: Automatically optimizes models for target hardware
- **Benefits**: Up to 2x performance improvement
- **Supported Frameworks**: TensorFlow, PyTorch, MXNet, ONNX

#### SageMaker Inference Recommender
- **Description**: Benchmarks different instance types and configurations
- **Benefits**: Helps identify optimal price-performance ratio

#### SageMaker Endpoints
- **Serverless Inference**: Pay-per-use model for sporadic workloads
- **Asynchronous Inference**: For non-real-time batch inference
- **Multi-Model Endpoints**: Host multiple models on single endpoint

#### ONNX Runtime
- **Description**: Cross-platform inference acceleration
- **Benefits**: Up to 17x faster inference for transformer models
- **Integration**: SageMaker containers with ONNX Runtime

#### AWS Inferentia/Trainium
- **Description**: Custom AWS silicon for ML inference/training
- **Benefits**: Up to 2.3x better price performance than GPU instances
- **Integration**: SageMaker Inf1/Trn1 instances

## Cost Analysis

Analysis of cost savings from different optimization techniques.

### Research Notes:
#### Baseline Costs (Unoptimized)
- **Model Size**: 8B parameter model in FP32 = ~32GB
- **Instance Types**:
  - ml.g5.2xlarge: $1.515/hour
  - ml.g5.12xlarge: $7.596/hour
- **Monthly Cost** (24/7 deployment):
  - ml.g5.2xlarge: ~$1,100/month
  - ml.g5.12xlarge: ~$5,500/month

#### Optimization Savings
- **INT8 Quantization**: ~4x reduction
  - Storage: 32GB → 8GB
  - Instance: Can potentially use smaller instance
  - Estimated savings: 30-50% on instance costs
  
- **Knowledge Distillation**: Model-dependent
  - Example: 8B → 1.1B parameters (TinyLlama)
  - Storage: 32GB → 4.4GB
  - Estimated savings: 50-70% on instance costs
  
- **Pruning + INT8**: ~6-8x reduction
  - Storage: 32GB → 4-5GB
  - Estimated savings: 40-60% on instance costs

#### Optimization Costs
- **Quantization**:
  - PTQ: 1-2 hours on ml.g5.2xlarge (~$3)
  - QAT: 1-2 days on ml.g5.12xlarge (~$180-$360)
  
- **Knowledge Distillation**:
  - Training: 3-7 days on ml.g5.12xlarge (~$550-$1,300)
  
- **Pruning**:
  - Fine-tuning: 2-4 days on ml.g5.12xlarge (~$360-$730)

#### ROI Analysis
- **PTQ**: Immediate ROI (hours to days)
- **Knowledge Distillation**: ROI in 1-3 months
- **Pruning**: ROI in 1-2 months
- **Combined Approaches**: Fastest ROI

## Implementation Examples

Code examples and implementation details.

### Research Notes:
#### PyTorch Quantization Example (PTQ)
```python
# Basic PyTorch quantization example for LLMs
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Quantize to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_llama_model.pth")
```

#### Hugging Face Optimum Example
```python
# Using Hugging Face Optimum for quantization
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Configure quantization
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# Quantize the model
quantizer.quantize(
    save_dir="./quantized_model",
    quantization_config=qconfig
)
```

#### TensorFlow Quantization Example
```python
# TensorFlow quantization example
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load model (example with smaller BERT for illustration)
model = tf.keras.models.load_model("bert_model")

# Apply quantization aware training
quantized_model = tfmot.quantization.keras.quantize_model(model)

# Fine-tune if needed
quantized_model.compile(...)
quantized_model.fit(...)

# Convert to TFLite for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)
```

#### SageMaker Deployment Example
```python
# SageMaker deployment of optimized model
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create PyTorch model
model = PyTorchModel(
    model_data="s3://bucket/path/to/optimized_model.tar.gz",
    role=role,
    framework_version="1.13.1",
    py_version="py39",
    entry_point="inference.py"
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="optimized-llm-endpoint"
)
```

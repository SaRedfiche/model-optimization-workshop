# Quantization Deep Dive for LLMs

This document provides a detailed exploration of quantization techniques specifically for large language models (LLMs) like Llama, BERT, GPT, and DeepSeek.

## Quantization Overview

Quantization is one of the most effective techniques for reducing LLM footprint with minimal performance degradation. It works by representing model weights and/or activations with lower precision numerical formats.

## Precision Formats

| Format | Bits | Size Reduction | Accuracy Impact | Notes |
|--------|------|----------------|-----------------|-------|
| FP32 | 32-bit | Baseline | None (reference) | Standard training format |
| FP16 | 16-bit | ~2x | Minimal | Good balance for most LLMs |
| BF16 | 16-bit | ~2x | Minimal | Better numerical stability than FP16 |
| INT8 | 8-bit | ~4x | Low-Moderate | Standard for production deployment |
| INT4 | 4-bit | ~8x | Moderate | Emerging standard for LLMs |
| INT2/3 | 2/3-bit | ~10-16x | High | Experimental, significant degradation |
| GGUF | Mixed | ~4-8x | Varies | Format used by llama.cpp |

## LLM-Specific Quantization Techniques

### GPTQ
- **Description**: Post-training quantization method specifically designed for LLMs
- **Precision**: INT4/INT8
- **Size Reduction**: ~8x with INT4
- **Performance Impact**: Minimal (1-2% degradation on most benchmarks)
- **Implementation**: 
  - AutoGPTQ library for PyTorch
  - Hugging Face Transformers integration
- **SageMaker Compatibility**: Requires custom container

### AWQ (Activation-aware Weight Quantization)
- **Description**: Optimizes quantization by considering activation patterns
- **Precision**: INT4/INT8
- **Size Reduction**: ~8x with INT4
- **Performance Impact**: Very low (often <1% degradation)
- **Implementation**:
  - AWQ library for PyTorch
  - Hugging Face Transformers integration
- **SageMaker Compatibility**: Requires custom container

### SmoothQuant
- **Description**: Balances quantization between activations and weights
- **Precision**: INT8
- **Size Reduction**: ~4x
- **Performance Impact**: Low
- **Implementation**: 
  - Available in NVIDIA TensorRT-LLM
- **SageMaker Compatibility**: Works with SageMaker TensorRT containers

### QLoRA
- **Description**: Combines quantization with parameter-efficient fine-tuning
- **Precision**: INT4/INT8 for base model, FP16 for adapters
- **Size Reduction**: ~8x for base model
- **Performance Impact**: Minimal for fine-tuned tasks
- **Implementation**:
  - PEFT library from Hugging Face
- **SageMaker Compatibility**: Works with standard SageMaker PyTorch containers
## Framework-Specific Implementation

### PyTorch

#### Native PyTorch Quantization
```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Dynamic quantization (easiest approach)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_llama_model.pth")
```

#### Hugging Face Optimum
```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoModelForCausalLM

# Load model
model_id = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(model_id)

# Configure quantization
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# Quantize the model
quantizer.quantize(
    save_dir="./quantized_model",
    quantization_config=qconfig
)
```

#### GPTQ Implementation
```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Define quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load and quantize model
model_id = "meta-llama/Llama-2-7b"
model = AutoGPTQForCausalLM.from_pretrained(model_id)
model.quantize(quantize_config)

# Save quantized model
model.save_quantized("./gptq_quantized_model")
```

### TensorFlow

#### TF-Lite Quantization
```python
import tensorflow as tf

# Load model (example with smaller BERT for illustration)
model = tf.saved_model.load("bert_model_path")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model("bert_model_path")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # For FP16 quantization
tflite_model = converter.convert()

# Save the model
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)
```

#### TensorFlow Model Optimization Toolkit
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load model
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
## SageMaker Deployment

### Deploying Quantized PyTorch Models
```python
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create PyTorch model
model = PyTorchModel(
    model_data="s3://bucket/path/to/quantized_model.tar.gz",
    role=role,
    framework_version="1.13.1",
    py_version="py39",
    entry_point="inference.py"
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="quantized-llm-endpoint"
)
```

### Custom Container for Advanced Quantization
```dockerfile
# Dockerfile for custom quantization container
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install dependencies
RUN pip install transformers optimum auto-gptq accelerate

# Copy inference code
COPY inference.py /opt/ml/code/inference.py

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
```

## Performance Benchmarks

| Model | Original Size | Quantized Size | Technique | Perplexity Change | Latency Impact |
|-------|---------------|----------------|-----------|-------------------|----------------|
| Llama-2-7B | 13GB | 3.5GB | GPTQ (INT4) | +0.2 | -5% |
| Llama-2-7B | 13GB | 7GB | INT8 | +0.1 | -15% |
| BERT-Large | 1.3GB | 330MB | INT8 | +0.05 | -20% |
| GPT-J-6B | 12GB | 3GB | GPTQ (INT4) | +0.3 | -10% |
| DeepSeek-7B | 14GB | 3.5GB | AWQ (INT4) | +0.15 | -5% |

## Cost Analysis

### Optimization Costs
- **GPTQ Quantization**:
  - Hardware: ml.g5.2xlarge ($1.515/hour)
  - Time: 2-4 hours
  - Total: $3-$6
  
- **AWQ Quantization**:
  - Hardware: ml.g5.2xlarge ($1.515/hour)
  - Time: 3-5 hours
  - Total: $4.50-$7.50

### Hosting Cost Comparison
- **Original 7B Model (FP16)**:
  - Instance: ml.g5.2xlarge ($1.515/hour)
  - Monthly cost (24/7): ~$1,100
  
- **Quantized 7B Model (INT4)**:
  - Instance: ml.g5.xlarge ($0.7575/hour)
  - Monthly cost (24/7): ~$550
  - Savings: ~50%

## Conclusion

Quantization offers one of the best returns on investment for optimizing LLM deployment costs. For most use cases, INT8 quantization provides a good balance between model size reduction and performance preservation. For more aggressive optimization, INT4 techniques like GPTQ and AWQ can reduce model size by up to 8x with minimal performance impact.

The choice of quantization technique should be based on:
1. Performance requirements
2. Model architecture
3. Framework compatibility
4. Deployment environment

For AWS SageMaker specifically, Hugging Face Optimum and ONNX Runtime integration provide the most straightforward path to deployment, while custom containers may be needed for more advanced techniques like GPTQ and AWQ.

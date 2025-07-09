# Standardized Imports for Model Optimization Workshop

This document explains the standardized import system for the Model Optimization Workshop notebooks.

## Overview

To simplify dependencies and avoid conflicts across notebooks, we've implemented a standardized import system:

1. **Unified Requirements**: All dependencies are listed in a single `requirements.txt` file
2. **Common Imports**: Frequently used imports are centralized in `common_imports.py`
3. **Workshop Configuration**: Shared configuration is stored in `workshop_config.json`

## How to Use

### In Notebook 1 (Introduction and Setup)

The first notebook installs all dependencies and creates the workshop configuration:

```python
# Install core dependencies first
!pip install -q "numpy>=1.23.0" "pandas>=1.5.3" "matplotlib>=3.6.3" "seaborn>=0.12.2"

# Install PyTorch
!pip install -q "torch==2.0.0" "torchvision==0.15.1" "torchaudio==2.0.1"

# Install transformers and related libraries
!pip install -q "transformers==4.26.0" "datasets==2.10.1" "accelerate==0.18.0"

# Install AWS libraries
!pip install -q "boto3>=1.35.0" "sagemaker>=2.130.0"

# Import common modules
from common_imports import *

# Create workshop configuration
workshop_config = {
    "base_model": "distilbert-base-uncased-finetuned-sst-2-english",
    "task": "sequence-classification",
    "s3_bucket": bucket,
    "s3_prefix": prefix,
    "region": region,
    "role": role,
    "device_type": DEVICE_INFO["type"],
    "created_at": time.strftime("%Y-%m-%d-%H-%M-%S")
}

# Save configuration to file
with open("workshop_config.json", "w") as f:
    json.dump(workshop_config, f, indent=2)
```

### In Subsequent Notebooks

In all other notebooks, use this pattern:

```python
# Import common modules
from common_imports import *

# Import specific modules for this notebook
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor

# Load workshop configuration
with open('workshop_config.json', 'r') as f:
    workshop_config = json.load(f)

# Set up SageMaker session
sagemaker_session = sagemaker.Session()
role = workshop_config['role']
region = workshop_config['region']
bucket = workshop_config['s3_bucket']
prefix = workshop_config['s3_prefix']
```

## What's Included in Common Imports

The `common_imports.py` file includes:

- Standard libraries (os, sys, time, json, etc.)
- Data processing (numpy, pandas)
- Visualization (matplotlib, seaborn)
- Machine learning (torch, transformers)
- AWS (boto3, sagemaker)
- Local utilities (utils.py, optimization_utils.py)
- Device detection (CPU/GPU)

## Handling Dependency Conflicts

The common_imports.py file is designed to handle potential dependency conflicts:

- Uses try/except blocks to gracefully handle missing packages
- Provides fallbacks when certain libraries aren't available
- Gives clear warnings when dependencies are missing

## Troubleshooting Common Issues

### NumPy C-API Version Mismatch

If you see an error like:
```
RuntimeError: module was compiled against NumPy C-API version 0x10 (NumPy 1.23) but the running NumPy has C-API version 0xf
```

Try these solutions:
1. Install the specific NumPy version required:
   ```
   pip install "numpy==1.23.0"
   ```
2. Reinstall packages that depend on NumPy:
   ```
   pip uninstall -y torch torchvision torchaudio
   pip install "torch==2.0.0" "torchvision==0.15.1" "torchaudio==2.0.1"
   ```

### Shell Redirection Issues

If you see files being created with names like `0.12.2` when running pip install commands, it's because the shell is interpreting `>=` as redirection operators. Always quote package specifications:

```
# Incorrect (creates files named 1.23.0, 1.5.3, etc.)
pip install numpy>=1.23.0 pandas>=1.5.3

# Correct (properly installs packages)
pip install "numpy>=1.23.0" "pandas>=1.5.3"
```

### Missing Transformers Library

If you see an error like:
```
NameError: name 'transformers' is not defined
```

Try these solutions:
1. Install transformers explicitly:
   ```
   pip install transformers==4.26.0
   ```
2. Check for import errors in the common_imports.py file:
   ```python
   try:
       import transformers
       print(f"Transformers version: {transformers.__version__}")
   except Exception as e:
       print(f"Error importing transformers: {e}")
   ```

### SageMaker Import Issues

If you have issues with SageMaker imports:
1. Install SageMaker explicitly:
   ```
   pip install sagemaker
   ```
2. Try importing specific components directly:
   ```python
   from sagemaker.processing import ProcessingInput, ProcessingOutput
   from sagemaker.pytorch.processing import PyTorchProcessor
   ```

## Benefits

- **Consistency**: Same import patterns across all notebooks
- **Maintainability**: Easier to update dependencies in one place
- **Reduced Conflicts**: Avoids version conflicts between notebooks
- **Simplified Setup**: One-time installation of all dependencies
- **Graceful Degradation**: Handles missing dependencies gracefully

## Fallback Strategy

If you continue to have issues with the standardized imports:

1. Use direct imports in each notebook instead of common_imports.py
2. Install only the packages needed for each specific notebook
3. Use the workshop_config.json file for configuration sharing

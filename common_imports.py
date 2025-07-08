"""
Common imports for the Model Optimization Workshop.
This file provides standardized imports for all notebooks.
"""

# Standard libraries
import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Data processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    Trainer, 
    TrainingArguments
)

# AWS
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# Utilities
from tqdm.auto import tqdm

# Local utilities
from utils import (
    measure_inference_time,
    get_model_size,
    plot_comparison,
    estimate_monthly_cost,
    load_model_and_tokenizer,
    prepare_inputs
)

from optimization_utils import (
    analyze_job_failure,
    save_checkpoint,
    handle_processing_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for GPU availability
def get_device_info():
    """Get information about the available device (CPU/GPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        return {
            "device": device,
            "name": f"GPU: {device_name}",
            "count": device_count,
            "type": "GPU"
        }
    else:
        device = torch.device("cpu")
        return {
            "device": device,
            "name": "CPU",
            "count": 1,
            "type": "CPU"
        }

# Get device information
DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["device"]

# Print device information
print(f"Using device: {DEVICE_INFO['name']} ({DEVICE_INFO['type']})")

# Set default figure size for plots
plt.rcParams["figure.figsize"] = (12, 8)

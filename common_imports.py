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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize availability flags
NUMPY_AVAILABLE = False
PANDAS_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
SEABORN_AVAILABLE = False
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
AWS_AVAILABLE = False

# Data processing - with careful error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NumPy import error: {e}")
    # Create a minimal numpy substitute for basic functionality
    class NumpySubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
    np = NumpySubstitute()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pandas import error: {e}")
    # Create a minimal pandas substitute
    class PandasSubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
    pd = PandasSubstitute()

# Visualization - with careful error handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    # Set default figure size for plots
    plt.rcParams["figure.figsize"] = (12, 8)
except ImportError as e:
    logger.warning(f"Matplotlib import error: {e}")
    # Create a minimal matplotlib substitute
    class MatplotlibSubstitute:
        class pyplot:
            def __init__(self):
                self.matplotlib = type('obj', (object,), {'__version__': "Not Available"})
            def figure(self, *args, **kwargs):
                logger.warning("Matplotlib not available. Figure creation skipped.")
    plt = MatplotlibSubstitute().pyplot()

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Seaborn import error: {e}")
    # Create a minimal seaborn substitute
    class SeabornSubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
        def barplot(self, *args, **kwargs):
            logger.warning("Seaborn not available. Plotting skipped.")
    sns = SeabornSubstitute()

# Machine learning - with careful error handling
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch import error: {e}")
    # Create a minimal torch substitute
    class TorchSubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
            self.cuda = type('obj', (object,), {'is_available': lambda: False})
            self.device = type('obj', (object,), {'__str__': lambda x: "cpu"})
    torch = TorchSubstitute()

# Transformers - with careful error handling
try:
    import transformers
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
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers import error: {e}")
    # Create a minimal transformers substitute
    class TransformersSubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
    transformers = TransformersSubstitute()
    # Create minimal substitutes for commonly used classes
    AutoTokenizer = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoModelForTokenClassification = None
    AutoModelForQuestionAnswering = None
    AutoModelForMaskedLM = None
    Trainer = None
    TrainingArguments = None

# AWS - with careful error handling
try:
    import boto3
    import sagemaker
    AWS_AVAILABLE = True
    try:
        from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
    except ImportError:
        logger.warning("SageMaker processing modules not available")
except ImportError as e:
    logger.warning(f"AWS libraries import error: {e}")
    # Create minimal substitutes
    class Boto3Substitute:
        def __init__(self):
            self.__version__ = "Not Available"
    boto3 = Boto3Substitute()
    
    class SagemakerSubstitute:
        def __init__(self):
            self.__version__ = "Not Available"
    sagemaker = SagemakerSubstitute()

# Utilities - with careful error handling
try:
    from tqdm.auto import tqdm
except ImportError:
    # Create a minimal tqdm substitute
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Check for GPU availability
def get_device_info():
    """Get information about the available device (CPU/GPU)."""
    if not TORCH_AVAILABLE:
        return {
            "device": "cpu",
            "name": "CPU (PyTorch not available)",
            "count": 1,
            "type": "CPU"
        }
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
        except:
            device_name = "Unknown GPU"
            device_count = 1
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

# Try to import local utilities, but don't fail if they're not available
try:
    from utils import (
        measure_inference_time,
        get_model_size,
        plot_comparison,
        estimate_monthly_cost,
        load_model_and_tokenizer,
        prepare_inputs
    )
except ImportError as e:
    logger.warning(f"utils.py import error: {e}")
    # Create minimal substitutes for commonly used functions
    def measure_inference_time(*args, **kwargs):
        logger.warning("measure_inference_time not available")
        return 0.0
    
    def get_model_size(*args, **kwargs):
        logger.warning("get_model_size not available")
        return 0.0
    
    def plot_comparison(*args, **kwargs):
        logger.warning("plot_comparison not available")
        return None
    
    def estimate_monthly_cost(*args, **kwargs):
        logger.warning("estimate_monthly_cost not available")
        return {"compute_cost": 0.0, "storage_cost": 0.0, "total_cost": 0.0}
    
    def load_model_and_tokenizer(*args, **kwargs):
        logger.warning("load_model_and_tokenizer not available")
        return None, None
    
    def prepare_inputs(*args, **kwargs):
        logger.warning("prepare_inputs not available")
        return {}

try:
    from optimization_utils import (
        analyze_job_failure,
        save_checkpoint,
        handle_processing_error
    )
except ImportError as e:
    logger.warning(f"optimization_utils.py import error: {e}")
    # Create minimal substitutes
    def analyze_job_failure(*args, **kwargs):
        logger.warning("analyze_job_failure not available")
        return "Function not available"
    
    def save_checkpoint(*args, **kwargs):
        logger.warning("save_checkpoint not available")
        return None
    
    def handle_processing_error(*args, **kwargs):
        logger.warning("handle_processing_error not available")
        return None

# Get device information
DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["device"]

# Print device information
print(f"Using device: {DEVICE_INFO['name']} ({DEVICE_INFO['type']})")

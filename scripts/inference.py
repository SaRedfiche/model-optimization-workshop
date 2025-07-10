#!/usr/bin/env python3
"""
Custom inference script for quantized models.
This script handles inference for ONNX quantized models in SageMaker endpoints.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir: str):
    """
    Load the model for inference.
    
    Args:
        model_dir: The directory where the model artifacts are stored.
        
    Returns:
        The loaded model and tokenizer.
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Check what type of model we have in the directory
    model_files = os.listdir(model_dir)
    has_onnx = any(f.endswith('.onnx') for f in model_files)
    has_pytorch = any(f.endswith(('.bin', '.safetensors')) for f in model_files)
    
    if not has_onnx and not has_pytorch:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    if has_onnx:
        # Load ONNX model
        logger.info("Loading ONNX quantized model...")
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        
        model = ORTModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info("Successfully loaded ONNX quantized model")
        return {"model": model, "tokenizer": tokenizer, "type": "onnx"}
        
    elif has_pytorch:
        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info("Successfully loaded PyTorch model")
        return {"model": model, "tokenizer": tokenizer, "type": "pytorch"}
    
    else:
        raise ValueError(f"Unsupported model format in {model_dir}")

def input_fn(request_body: str, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Parse input data for inference.
    
    Args:
        request_body: The request body containing input data.
        content_type: The content type of the request.
        
    Returns:
        Parsed input data.
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    if content_type == "application/json":
        input_data = json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    # Handle different input formats
    if isinstance(input_data, dict):
        if "inputs" in input_data:
            texts = input_data["inputs"]
        elif "text" in input_data:
            texts = input_data["text"]
        else:
            texts = input_data
    elif isinstance(input_data, (list, str)):
        texts = input_data
    else:
        raise ValueError(f"Unsupported input format: {type(input_data)}")
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    return {"texts": texts}

def predict_fn(input_data: Dict[str, Any], model_artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run inference on the input data.
    
    Args:
        input_data: The parsed input data.
        model_artifacts: The loaded model and tokenizer.
        
    Returns:
        Prediction results.
    """
    texts = input_data["texts"]
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    model_type = model_artifacts["type"]
    
    logger.info(f"Running inference on {len(texts)} text(s) using {model_type} model")
    
    try:
        # Tokenize inputs
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad() if model_type == "pytorch" else nullcontext():
            outputs = model(**inputs)
        
        # Process outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]
        
        # Apply softmax to get probabilities
        import torch
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predictions
        predictions = []
        for i, probs in enumerate(probabilities):
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
            
            # Get label from model config - fail if not available
            if hasattr(model.config, 'id2label') and model.config.id2label:
                label = model.config.id2label[pred_id]
            else:
                raise ValueError(f"Model config missing id2label mapping. Cannot determine label for prediction ID {pred_id}")
            
            predictions.append({
                "label": label,
                "score": confidence
            })
        
        logger.info(f"Successfully processed {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def output_fn(predictions: List[Dict[str, Any]], accept: str = "application/json") -> str:
    """
    Format the prediction output.
    
    Args:
        predictions: The prediction results.
        accept: The requested output format.
        
    Returns:
        Formatted output.
    """
    logger.info(f"Formatting output with accept type: {accept}")
    
    if accept == "application/json":
        return json.dumps(predictions)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

# Context manager for non-PyTorch models
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Import torch at module level for compatibility
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available, some functionality may be limited")
    torch = None

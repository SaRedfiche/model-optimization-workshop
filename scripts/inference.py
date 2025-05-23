#!/usr/bin/env python3
"""
Inference script for SageMaker endpoint deployment.
This script loads a quantized ONNX model and performs inference.
"""

import os
import json
import logging
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variables
tokenizer = None
ort_session = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def model_fn(model_dir):
    """
    Load the model and tokenizer for inference.
    
    Args:
        model_dir (str): The directory where model artifacts are stored.
        
    Returns:
        dict: Dictionary containing the loaded model and tokenizer.
    """
    global tokenizer, ort_session
    
    logger.info(f"Loading model from {model_dir}")
    
    # List contents of model directory
    logger.info(f"Contents of model directory: {os.listdir(model_dir)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load ONNX model
    model_path = os.path.join(model_dir, "model.onnx")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(model_path)
    
    logger.info("Model loaded successfully")
    return {"tokenizer": tokenizer, "model": ort_session}

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body (str): The request body.
        request_content_type (str): The request content type.
        
    Returns:
        dict: The prepared input for prediction.
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Check if input is in the expected format
        if not isinstance(input_data, dict) or "inputs" not in input_data:
            input_data = {"inputs": input_data}
        
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate a prediction for the input data.
    
    Args:
        input_data (dict): The input data.
        model (dict): The loaded model and tokenizer.
        
    Returns:
        list: The prediction results.
    """
    global tokenizer, ort_session
    
    # Extract inputs
    inputs = input_data["inputs"]
    
    # Tokenize inputs
    encoded_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        "input_ids": encoded_inputs["input_ids"].numpy(),
        "attention_mask": encoded_inputs["attention_mask"].numpy()
    }
    
    # Add token_type_ids if present in the tokenizer output
    if "token_type_ids" in encoded_inputs:
        ort_inputs["token_type_ids"] = encoded_inputs["token_type_ids"].numpy()
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Process outputs
    logits = ort_outputs[0]
    predictions = np.argmax(logits, axis=1).tolist()
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy().tolist()
    
    # Prepare results
    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "label": pred,
            "score": max(prob),
            "probabilities": prob
        })
    
    return results

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    
    Args:
        prediction (list): The prediction result.
        response_content_type (str): The response content type.
        
    Returns:
        str: The serialized prediction.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model and tokenizer"""
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # Find the actual model directory (may be nested)
        import os
        model_path = model_dir
        for root, dirs, files in os.walk(model_dir):
            if any(f.endswith(('.bin', '.safetensors')) for f in files) and 'config.json' in files:
                model_path = root
                break
        
        logger.info(f"Using model path: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        logger.info("Model and tokenizer loaded successfully")
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Handle different input formats
        if isinstance(input_data, dict):
            if "inputs" in input_data:
                return input_data["inputs"]
            elif "text" in input_data:
                return input_data["text"]
            else:
                return input_data
        else:
            return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Handle both single string and list of strings
    if isinstance(input_data, str):
        texts = [input_data]
        single_input = True
    else:
        texts = input_data
        single_input = False
    
    # Tokenize input
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convert to list and return
    results = []
    for i, pred in enumerate(predictions):
        # Get the predicted class and confidence
        predicted_class = torch.argmax(pred).item()
        confidence = float(torch.max(pred))
        
        # Map to sentiment labels (0=NEGATIVE, 1=POSITIVE for SST-2)
        label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
        
        result = {
            "label": label,
            "score": confidence,
            "scores": {
                "NEGATIVE": float(pred[0]),
                "POSITIVE": float(pred[1])
            }
        }
        results.append(result)
    
    return results[0] if single_input else results

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

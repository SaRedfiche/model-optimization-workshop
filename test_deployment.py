#!/usr/bin/env python3
"""
Test deployment script for the fine-tuned sentiment analysis model.
This script deploys the model using PyTorch container and tests it.
"""

import os
import json
import time
import logging
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_inference_script():
    """Create a simple inference script for sentiment analysis"""
    inference_script = '''
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
'''
    
    # Save inference script
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/sentiment_inference.py", "w") as f:
        f.write(inference_script)
    
    logger.info("Sentiment inference script created successfully!")

def create_requirements_file():
    """Create requirements file for deployment"""
    requirements = '''transformers>=4.49.0
torch>=2.0.0
numpy>=1.26.0
'''
    
    with open("scripts/requirements.txt", "w") as f:
        f.write(requirements)
    
    logger.info("Requirements file created successfully!")

def deploy_model(model_data_uri):
    """Deploy the fine-tuned model using PyTorch container"""
    
    # Get execution role
    role = sagemaker.get_execution_role()
    logger.info(f"Using SageMaker execution role: {role}")
    
    # Create inference script and requirements
    create_inference_script()
    create_requirements_file()
    
    # Create PyTorch model for deployment
    logger.info("Creating PyTorch model for deployment...")
    pytorch_model = PyTorchModel(
        model_data=model_data_uri,
        role=role,
        framework_version="2.5.1",  # Latest supported PyTorch version
        py_version="py311",         # Modern Python version
        entry_point="sentiment_inference.py",
        source_dir="./scripts",
        env={
            'HF_TASK': 'text-classification',
            'TRANSFORMERS_CACHE': '/tmp/transformers_cache'
        }
    )
    
    logger.info("PyTorch model created successfully!")
    
    # Deploy model to endpoint
    endpoint_name = f"sentiment-analysis-{int(time.time())}"
    logger.info(f"Deploying model to endpoint: {endpoint_name}")
    
    try:
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge',  # CPU instance for cost efficiency
            endpoint_name=endpoint_name,
            wait=True
        )
        
        logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
        return predictor, endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

def test_model(predictor):
    """Test the deployed model with sample texts"""
    
    # Set serializers
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    
    # Test samples
    sample_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "This movie was terrible. I hated it.",
        "The movie was okay, nothing special.",
        "Amazing performance by the actors!",
        "Boring and predictable plot."
    ]
    
    logger.info("Testing model with sample texts...")
    print("\n" + "="*60)
    print("MODEL TESTING RESULTS")
    print("="*60)
    
    for i, text in enumerate(sample_texts, 1):
        try:
            start_time = time.time()
            response = predictor.predict({"inputs": text})
            end_time = time.time()
            
            print(f"\nTest {i}:")
            print(f"Text: {text}")
            print(f"Prediction: {response['label']} (confidence: {response['score']:.4f})")
            print(f"Scores: NEGATIVE={response['scores']['NEGATIVE']:.4f}, POSITIVE={response['scores']['POSITIVE']:.4f}")
            print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error testing text '{text}': {e}")
    
    print("\n" + "="*60)
    
    # Test batch prediction
    logger.info("Testing batch prediction...")
    try:
        start_time = time.time()
        batch_response = predictor.predict({"inputs": sample_texts[:3]})
        end_time = time.time()
        
        print(f"\nBatch Prediction Test:")
        print(f"Input: {len(sample_texts[:3])} texts")
        print(f"Results: {len(batch_response)} predictions")
        print(f"Batch inference time: {(end_time - start_time) * 1000:.2f} ms")
        
        for i, (text, result) in enumerate(zip(sample_texts[:3], batch_response)):
            print(f"  {i+1}. '{text[:30]}...' -> {result['label']} ({result['score']:.4f})")
            
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")

def main():
    """Main deployment and testing function"""
    
    # Model data URI from the training job
    model_data_uri = "s3://model-optimization-test-1752261534/sentiment-analysis/output/huggingface-pytorch-training-2025-07-11-22-35-09-918/output/model.tar.gz"
    
    logger.info("Starting model deployment test...")
    logger.info(f"Model data URI: {model_data_uri}")
    
    try:
        # Deploy the model
        predictor, endpoint_name = deploy_model(model_data_uri)
        
        # Test the model
        test_model(predictor)
        
        # Ask user if they want to keep the endpoint
        print(f"\n{'='*60}")
        print("DEPLOYMENT SUCCESSFUL!")
        print(f"Endpoint name: {endpoint_name}")
        print("The model is now deployed and ready for use.")
        print(f"{'='*60}")
        
        # Note: We'll leave the endpoint running for now
        # User can delete it manually or we can add cleanup later
        logger.info("Deployment test completed successfully!")
        
        return predictor, endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment test failed: {e}")
        raise

if __name__ == "__main__":
    main()

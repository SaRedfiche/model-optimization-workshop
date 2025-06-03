from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model(model_path, text):
    """Test a model on a given text"""
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Process text
    logger.info(f"Processing text: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    # Get predictions
    predictions = outputs.logits.argmax(dim=2)
    
    # Convert predictions to labels
    id2label = model.config.id2label
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Print results
    logger.info("Results:")
    for token, pred in zip(tokens, predictions[0].tolist()):
        label = id2label[pred]
        logger.info(f"{token} -> {label}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Test sentence
    test_sentence = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
    
    # Test pruned model
    pruned_model_path = "output/ner_pruned_light"
    test_model(pruned_model_path, test_sentence)

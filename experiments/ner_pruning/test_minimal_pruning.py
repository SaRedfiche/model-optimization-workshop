import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_entities(tokens, predictions, id2label):
    """Extract entities from token predictions with proper handling of B- and I- tags"""
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        if token in ("[CLS]", "[SEP]"):
            continue
            
        label = id2label[pred]
        
        # Handle B- tags (beginning of entity)
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]  # Remove B- prefix
            current_entity = {"entity": entity_type, "text": token.replace("##", "")}
        
        # Handle I- tags (inside of entity)
        elif label.startswith("I-"):
            entity_type = label[2:]
            # If we have a current entity of the same type, append to it
            if current_entity and current_entity["entity"] == entity_type:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]  # Remove ## prefix
                else:
                    current_entity["text"] += " " + token
            # If we don't have a current entity or it's of a different type, treat as B-
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"entity": entity_type, "text": token.replace("##", "")}
        
        # Handle O tags (outside of entity)
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add the last entity if there is one
    if current_entity:
        entities.append(current_entity)
    
    return entities

def test_ner(model, tokenizer, text):
    """Test NER on a given text with proper entity extraction"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Convert predictions to labels
    id2label = model.config.id2label
    
    # Print token-level predictions
    logger.info("Token-level predictions:")
    for token, pred in zip(tokens, predictions):
        label = id2label[pred]
        logger.info(f"{token} -> {label}")
    
    # Extract entities
    entities = extract_entities(tokens, predictions, id2label)
    
    return entities

def main():
    # Test both models
    original_model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
    minimal_pruned_model_path = "output/ner_pruned_minimal"
    
    # Load original model
    logger.info(f"Loading original model from {original_model_path}")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    original_model = AutoModelForTokenClassification.from_pretrained(original_model_path)
    
    # Load minimally pruned model
    logger.info(f"Loading minimally pruned model from {minimal_pruned_model_path}")
    minimal_pruned_tokenizer = AutoTokenizer.from_pretrained(minimal_pruned_model_path)
    minimal_pruned_model = AutoModelForTokenClassification.from_pretrained(minimal_pruned_model_path)
    
    # Calculate sparsity for minimally pruned model
    param_size = sum(p.numel() for p in minimal_pruned_model.parameters())
    non_zero = sum(torch.count_nonzero(p).item() for p in minimal_pruned_model.parameters())
    sparsity = 100 * (1 - non_zero / param_size)
    logger.info(f"Minimally pruned model sparsity: {sparsity:.2f}%")
    
    # Test sentences
    test_sentences = [
        "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington.",
        "Apple Inc. was established by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
        "The European Union has its administrative headquarters in Brussels, Belgium.",
        "Barack Obama served as the 44th President of the United States from 2009 to 2017."
    ]
    
    original_total_entities = 0
    minimal_pruned_total_entities = 0
    
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n{'='*50}")
        logger.info(f"Test sentence {i+1}: {sentence}")
        logger.info(f"{'='*50}")
        
        logger.info("\nORIGINAL MODEL RESULTS:")
        original_entities = test_ner(original_model, original_tokenizer, sentence)
        logger.info(f"Entities found: {len(original_entities)}")
        for entity in original_entities:
            logger.info(f"  {entity['entity']}: {entity['text']}")
        original_total_entities += len(original_entities)
        
        logger.info("\nMINIMALLY PRUNED MODEL RESULTS:")
        minimal_pruned_entities = test_ner(minimal_pruned_model, minimal_pruned_tokenizer, sentence)
        logger.info(f"Entities found: {len(minimal_pruned_entities)}")
        for entity in minimal_pruned_entities:
            logger.info(f"  {entity['entity']}: {entity['text']}")
        minimal_pruned_total_entities += len(minimal_pruned_entities)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Original model total entities found: {original_total_entities}")
    logger.info(f"Minimally pruned model total entities found: {minimal_pruned_total_entities}")
    
    if original_total_entities > 0:
        entity_retention = (minimal_pruned_total_entities / original_total_entities) * 100
        logger.info(f"Entity retention rate: {entity_retention:.2f}%")

if __name__ == "__main__":
    main()

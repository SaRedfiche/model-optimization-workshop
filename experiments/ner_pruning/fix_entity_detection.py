import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ner(model, tokenizer, text):
    """Test NER on a given text with proper token alignment"""
    # Tokenize with word IDs to track original tokens
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True)
    offset_mapping = inputs.pop("offset_mapping")
    special_tokens_mask = inputs.pop("special_tokens_mask")
    
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
    for i, (token, pred, special) in enumerate(zip(tokens, predictions, special_tokens_mask[0].tolist())):
        if special == 1:  # Skip special tokens like [CLS] and [SEP]
            continue
        label = id2label[pred]
        logger.info(f"{token} -> {label}")
    
    # Extract entities with proper handling of subword tokens
    entities = []
    current_entity = None
    prev_token_idx = None
    
    for i, (token, pred, special, offset) in enumerate(zip(tokens, predictions, special_tokens_mask[0].tolist(), offset_mapping[0].tolist())):
        if special == 1:  # Skip special tokens
            continue
            
        label = id2label[pred]
        
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]  # Remove B- prefix
            current_entity = {"entity": entity_type, "text": token.replace("##", ""), "start": offset[0]}
            prev_token_idx = offset[1]
        
        elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
            # Check if this token is contiguous with the previous one
            if offset[0] == prev_token_idx:
                current_entity["text"] += token.replace("##", "")
            else:
                current_entity["text"] += " " + token.replace("##", "")
            prev_token_idx = offset[1]
        
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                prev_token_idx = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def main():
    # Test both models
    original_model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
    pruned_model_path = "output/ner_pruned_light"
    
    # Load original model
    logger.info(f"Loading original model from {original_model_path}")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    original_model = AutoModelForTokenClassification.from_pretrained(original_model_path)
    
    # Load pruned model
    logger.info(f"Loading pruned model from {pruned_model_path}")
    pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)
    pruned_model = AutoModelForTokenClassification.from_pretrained(pruned_model_path)
    
    # Test sentences
    test_sentences = [
        "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington.",
        "Apple Inc. was established by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
        "The European Union has its administrative headquarters in Brussels, Belgium.",
        "Barack Obama served as the 44th President of the United States from 2009 to 2017."
    ]
    
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n{'='*50}")
        logger.info(f"Test sentence {i+1}: {sentence}")
        logger.info(f"{'='*50}")
        
        logger.info("\nORIGINAL MODEL RESULTS:")
        original_entities = test_ner(original_model, original_tokenizer, sentence)
        logger.info(f"Entities found: {len(original_entities)}")
        for entity in original_entities:
            logger.info(f"  {entity['entity']}: {entity['text']}")
        
        logger.info("\nPRUNED MODEL RESULTS:")
        pruned_entities = test_ner(pruned_model, pruned_tokenizer, sentence)
        logger.info(f"Entities found: {len(pruned_entities)}")
        for entity in pruned_entities:
            logger.info(f"  {entity['entity']}: {entity['text']}")

if __name__ == "__main__":
    main()

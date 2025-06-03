import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load model and tokenizer from path"""
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    load_time = time.time() - start_time
    
    # Get model size
    param_size = sum(p.numel() for p in model.parameters())
    
    # Count non-zero parameters to calculate sparsity
    non_zero = sum(torch.count_nonzero(p).item() for p in model.parameters())
    sparsity = 100 * (1 - non_zero / param_size)
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Loading time: {load_time:.2f} seconds")
    logger.info(f"Model size: {param_size:,} parameters")
    logger.info(f"Non-zero parameters: {non_zero:,}")
    logger.info(f"Sparsity: {sparsity:.2f}%")
    
    return model, tokenizer

def test_ner(model, tokenizer, text):
    """Test NER on a given text"""
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        inference_time = time.time() - start_time
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert predictions to labels
    id2label = model.config.id2label
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Extract entities
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, predictions[0].tolist())):
        label = id2label[pred]
        
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": label[2:], "token": token, "start": i}
        elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
            current_entity["token"] += token.replace("##", "")
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities, inference_time

def main():
    # Original model
    original_model_path = "dbmdz/bert-large-cased-finetuned-conll03-english"
    logger.info(f"Loading original model from {original_model_path}")
    original_model, original_tokenizer = load_model(original_model_path)
    
    # Pruned model
    pruned_model_path = "output/ner_pruned_light"
    logger.info(f"Loading pruned model from {pruned_model_path}")
    pruned_model, pruned_tokenizer = load_model(pruned_model_path)
    
    # Test sentences
    test_sentences = [
        "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington.",
        "Apple Inc. was established by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
        "The European Union has its administrative headquarters in Brussels, Belgium.",
        "Barack Obama served as the 44th President of the United States from 2009 to 2017."
    ]
    
    logger.info("\n" + "="*50)
    logger.info("TESTING ORIGINAL MODEL")
    logger.info("="*50)
    
    original_total_time = 0
    original_entities = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\nTest sentence {i+1}: {sentence}")
        entities, inference_time = test_ner(original_model, original_tokenizer, sentence)
        original_total_time += inference_time
        original_entities.append(entities)
        
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        logger.info(f"Entities found: {len(entities)}")
        for entity in entities:
            logger.info(f"  {entity['entity']}: {entity['token']}")
    
    logger.info("\n" + "="*50)
    logger.info("TESTING PRUNED MODEL")
    logger.info("="*50)
    
    pruned_total_time = 0
    pruned_entities = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\nTest sentence {i+1}: {sentence}")
        entities, inference_time = test_ner(pruned_model, pruned_tokenizer, sentence)
        pruned_total_time += inference_time
        pruned_entities.append(entities)
        
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        logger.info(f"Entities found: {len(entities)}")
        for entity in entities:
            logger.info(f"  {entity['entity']}: {entity['token']}")
    
    # Performance comparison
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*50)
    logger.info(f"Original model total inference time: {original_total_time:.4f} seconds")
    logger.info(f"Pruned model total inference time: {pruned_total_time:.4f} seconds")
    
    if original_total_time > 0:
        speed_improvement = (original_total_time - pruned_total_time) / original_total_time * 100
        logger.info(f"Speed improvement: {speed_improvement:.2f}%")
    
    # Entity recognition comparison
    total_original_entities = sum(len(entities) for entities in original_entities)
    total_pruned_entities = sum(len(entities) for entities in pruned_entities)
    
    logger.info(f"Original model total entities found: {total_original_entities}")
    logger.info(f"Pruned model total entities found: {total_pruned_entities}")
    
    if total_original_entities > 0:
        entity_retention = (total_pruned_entities / total_original_entities) * 100
        logger.info(f"Entity retention rate: {entity_retention:.2f}%")

if __name__ == "__main__":
    main()

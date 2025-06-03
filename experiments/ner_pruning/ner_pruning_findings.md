# Special Considerations for NER Models

Through extensive experimentation, we've discovered that pruning is not suitable for Named Entity Recognition (NER) models based on transformer architectures like BERT. Our findings show that even minimal pruning significantly impacts entity recognition capabilities.

## NER Model Pruning Results

We tested three different pruning configurations on a BERT-based NER model:

1. **Heavy Pruning (30%)**: Complete failure - no entities detected
2. **Light Pruning (10%)**: Partial functionality - only location entities retained (33% of original entities)
3. **Minimal Pruning (3%)**: Maintained functionality but with errors - all entity types retained but with boundary issues

## Why Pruning Fails for NER Models

1. **Token-level Classification Sensitivity**: NER models make token-by-token predictions that are highly sensitive to contextual relationships between tokens
2. **Attention Mechanism Importance**: The attention mechanisms in transformer models are critical for capturing token relationships, and pruning disrupts these mechanisms
3. **Entity Type Sensitivity**: Different entity types (Person, Organization, Location) show varying levels of sensitivity to pruning
4. **Boundary Detection Issues**: Even with minimal pruning, entity boundary detection is significantly affected

## Recommendations for NER Models

- **Avoid Pruning**: Do not apply pruning to NER models
- **Alternative Approaches**: Consider quantization or knowledge distillation instead
- **Smaller Base Models**: If size reduction is necessary, start with a smaller pre-trained model rather than pruning a larger one

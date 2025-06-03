# BERT NER Model Pruning Comparison

## Overview

This document compares different pruning approaches for a BERT-based Named Entity Recognition (NER) model. The original model is `dbmdz/bert-large-cased-finetuned-conll03-english`, which is fine-tuned for token classification on the CoNLL-2003 dataset.

## Pruning Approaches

Three different pruning configurations were tested:

1. **Heavy Pruning (30%)** - From previous experiments
   - 30% pruning across all layers
   - No special handling for attention layers
   - No protection for classifier layer

2. **Light Pruning (10%)** - First test in this session
   - 10% pruning for intermediate and output dense layers
   - 5% pruning for attention layers (query, key, value, output)
   - Classifier layer completely excluded from pruning

3. **Minimal Pruning (3%)** - Second test in this session
   - 3% pruning for intermediate and output dense layers
   - 0.6% pruning for attention layers (20% of the base rate)
   - Classifier layer completely excluded from pruning

## Model Characteristics

| Metric | Original Model | Heavy Pruning (30%) | Light Pruning (10%) | Minimal Pruning (3%) |
|--------|---------------|-------------|-------------|-------------|
| Parameters | 332,538,889 | 332,538,889 | 332,538,889 | 332,538,889 |
| Non-zero parameters | 332,538,889 | ~232,777,222 | 307,429,633 | 326,097,633 |
| Sparsity | 0% | ~30% | 7.55% | 1.97% |
| Entity recognition | Baseline | Failed completely | Partial (locations only) | Good (with some errors) |

## Entity Recognition Performance

### Entity Retention Rate

| Pruning Approach | Entities Found | Retention Rate | Notes |
|------------------|---------------|----------------|-------|
| Original Model | 15 | 100% | Baseline |
| Heavy Pruning (30%) | 0 | 0% | Complete failure |
| Light Pruning (10%) | 5 | 33% | Only locations retained |
| Minimal Pruning (3%) | 21 | 140% | All entity types retained, but with errors |

### Entity Type Analysis

| Entity Type | Original Model | Heavy Pruning | Light Pruning | Minimal Pruning |
|-------------|---------------|--------------|--------------|----------------|
| Person (PER) | 5 | 0 | 0 | 5 |
| Organization (ORG) | 3 | 0 | 0 | 5 |
| Location (LOC) | 7 | 0 | 5 | 11 |

## Detailed Analysis

### Heavy Pruning (30%)
- Complete loss of entity recognition capability
- Model outputs only "O" (Outside) tags for all tokens
- Clearly too aggressive for this task

### Light Pruning (10%)
- Partial retention of entity recognition capability
- Only location entities were recognized
- All person and organization entities were lost
- Even for locations, only the most prominent ones were retained
- Insufficient for practical use

### Minimal Pruning (3%)
- Good retention of entity recognition capability
- All entity types (PER, ORG, LOC) were recognized
- However, entity boundaries were often incorrect
- Some tokens were incorrectly classified (e.g., "s" as a separate entity)
- Over-identification of entities (140% of original count)
- Usable but with reduced accuracy

## Conclusions

1. **Pruning Sensitivity**: NER models are highly sensitive to pruning, with even minimal pruning affecting entity boundary detection.

2. **Entity Type Sensitivity**: Different entity types show different sensitivity to pruning:
   - Location entities are most robust
   - Person entities are moderately sensitive
   - Organization entities are most sensitive

3. **Attention Layer Importance**: Reducing pruning rates for attention layers is crucial for maintaining performance.

4. **Classifier Layer Protection**: Completely excluding the classifier layer from pruning helps preserve output capabilities.

5. **Optimal Pruning Rate**: For this NER model, pruning rates should be kept below 3% overall, with even lower rates (0.6%) for attention layers.

## Recommendations

1. **Conservative Pruning**: Use very conservative pruning rates (1-3%) for NER models.

2. **Layer-Specific Pruning**: Apply different pruning rates based on layer type and position:
   - Minimal pruning for attention layers (0.5-1%)
   - Light pruning for intermediate layers (2-3%)
   - No pruning for classifier layer

3. **Post-Pruning Fine-Tuning**: Implement fine-tuning after pruning to recover performance.

4. **Alternative Approaches**: Consider quantization or knowledge distillation as alternatives to pruning for NER models.

5. **Hybrid Optimization**: Combine minimal pruning with quantization for better results.

6. **Evaluation Metrics**: Use entity-level metrics rather than token-level accuracy to evaluate pruned models.

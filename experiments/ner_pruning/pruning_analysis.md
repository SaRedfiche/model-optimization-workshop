# BERT NER Model Pruning Analysis

## Overview

This document analyzes the results of applying structured pruning to a BERT-based Named Entity Recognition (NER) model. The original model is `dbmdz/bert-large-cased-finetuned-conll03-english`, which is fine-tuned for token classification on the CoNLL-2003 dataset.

## Pruning Approach

The pruning was implemented with the following characteristics:

1. **Structured pruning**: Zeroing out entire neurons based on their L2 norm
2. **Selective pruning rates**:
   - 5% pruning for attention layers (query, key, value, output)
   - 10% pruning for intermediate and output dense layers
   - 0% pruning for the classifier layer (completely preserved)
3. **Overall sparsity**: 7.55% of parameters were zeroed out

## Model Characteristics

| Metric | Original Model | Pruned Model |
|--------|---------------|-------------|
| Parameters | 332,538,889 | 332,538,889 |
| Non-zero parameters | 332,538,889 | 307,429,633 |
| Sparsity | 0% | 7.55% |
| Loading time | 1.67s | 0.30s |
| Inference time (4 sentences) | 0.4998s | 0.4824s |
| Speed improvement | - | 3.48% |

## Entity Recognition Performance

### Test Sentence 1
"Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."

| Entity | Original Model | Pruned Model |
|--------|---------------|-------------|
| PER: Jeff Bezos | ✓ | ✗ |
| ORG: Amazon | ✓ | ✗ |
| LOC: Seattle | ✓ | ✓ |
| LOC: Washington | ✓ | ✓ |

### Test Sentence 2
"Apple Inc. was established by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California."

| Entity | Original Model | Pruned Model |
|--------|---------------|-------------|
| ORG: Apple Inc | ✓ | ✗ |
| PER: Steve Jobs | ✓ | ✗ |
| PER: Steve Wozniak | ✓ | ✗ |
| PER: Ronald Wayne | ✓ | ✗ |
| LOC: Cupertino | ✓ | ✗ |
| LOC: California | ✓ | ✓ |

### Test Sentence 3
"The European Union has its administrative headquarters in Brussels, Belgium."

| Entity | Original Model | Pruned Model |
|--------|---------------|-------------|
| ORG: European Union | ✓ | ✗ |
| LOC: Brussels | ✓ | ✗ |
| LOC: Belgium | ✓ | ✓ |

### Test Sentence 4
"Barack Obama served as the 44th President of the United States from 2009 to 2017."

| Entity | Original Model | Pruned Model |
|--------|---------------|-------------|
| PER: Barack Obama | ✓ | ✗ |
| LOC: United States | ✓ | ✓ |

## Analysis

1. **Entity Type Retention**:
   - The pruned model retained only location (LOC) entities
   - All person (PER) and organization (ORG) entities were lost
   - Even for locations, only the most prominent ones were retained

2. **Performance Impact**:
   - Original model identified 15 entities across all test sentences
   - Pruned model identified only 5 entities (33% retention rate)
   - The pruned model shows a clear bias toward location entities

3. **Speed Improvement**:
   - The pruning resulted in only a modest 3.48% speed improvement
   - This is not sufficient to justify the significant loss in entity recognition capability

## Conclusions

1. **Pruning Impact**: Even with conservative pruning rates (5-10%) and protecting the classifier layer, the model's ability to recognize entities was significantly degraded.

2. **Entity Type Sensitivity**: Different entity types showed different sensitivity to pruning:
   - Location entities were most robust
   - Person and organization entities were completely lost

3. **Recommendations**:
   - Further reduce pruning rates (perhaps to 2-3% for attention layers)
   - Consider alternative pruning approaches like magnitude pruning with a threshold
   - Explore knowledge distillation as an alternative to pruning
   - Implement fine-tuning after pruning to recover performance

4. **Trade-offs**: The current pruning approach creates a sparse model but doesn't reduce the model size. The minimal speed improvement doesn't justify the significant performance degradation.

5. **Next Steps**:
   - Test with even lower pruning rates
   - Implement post-pruning fine-tuning
   - Explore quantization as an alternative optimization approach
   - Consider hybrid approaches combining minimal pruning with quantization
